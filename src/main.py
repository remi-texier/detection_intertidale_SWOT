# src/main.py
import sys
import os
import multiprocessing
import threading
import logging
import traceback
import time
from collections import deque, defaultdict
from typing import Dict, List, Any, Tuple

from rich.console import Console, Group
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.table import Table

from . import config as app_config
from . import analysis
from . import plotting

console = Console()

log = logging.getLogger("rich_app")
log.setLevel(logging.INFO)
log.propagate = False

class QueueRelay:
    def __init__(self, queue, pid, log_type="log"):
        self.queue = queue
        self.pid = pid
        self.log_type = log_type
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            self.buffer = lines[-1]
            for line in lines[:-1]:
                line = line.strip()
                if line:
                    self.queue.put((self.pid, self.log_type, line))
    
    def flush(self):
        if self.buffer:
            line = self.buffer.strip()
            if line:
                self.queue.put((self.pid, self.log_type, line))
            self.buffer = ""

class WorkerState:
    def __init__(self, num_logs: int = 5):
        self.progress = Progress(TextColumn("[#5f87d7]Progression Tâche[/]"), BarColumn(), transient=True)
        self.task_id = self.progress.add_task("Démarrage...", total=100)
        self.logs = deque(maxlen=num_logs)
        self.title = "[grey50]Inactif..."
        self.is_active = False

    def get_renderable(self) -> Panel:
        log_text = Text("\n".join(self.logs), justify="left")
        render_group = Group(self.progress, log_text)
        return Panel(render_group, title=self.title, border_style="blue" if self.is_active else "grey50")

worker_states: Dict[int, WorkerState] = {}
state_lock = threading.Lock()
pid_to_worker_id: Dict[int, int] = {}
worker_last_completion_time: Dict[int, float] = {}  

def progress_listener(queue, num_workers):
    while True:
        try:
            message = queue.get()
            if message == "STOP": break
            pid, msg_type, content = message
            with state_lock:
                if pid not in pid_to_worker_id:
                    available_slot = -1
                    used_slots = set(pid_to_worker_id.values())
                    for i in range(num_workers):
                        if i not in used_slots and not worker_states[i].is_active:
                            if i in worker_last_completion_time:
                                time_since_completion = time.time() - worker_last_completion_time[i]
                                if time_since_completion < 3.0:
                                    continue 
                            available_slot = i
                            break
                    if available_slot != -1:
                        state = worker_states[available_slot]
                        state.title = "[grey50]Attribution..."
                        state.progress.reset(state.task_id)
                        state.logs.clear()
                        pid_to_worker_id[pid] = available_slot
                    else: continue
                worker_id = pid_to_worker_id[pid]
                state = worker_states[worker_id]
                if msg_type == "status":
                    if not state.is_active: state.is_active = True
                    state.title = content
                    if state.progress.tasks[0].completed < 95:
                        state.progress.update(state.task_id, advance=5)
                elif msg_type == "log":
                    state.logs.append(content)
                elif msg_type == "final":
                    state.title = content
                    state.progress.update(state.task_id, completed=100)
                    state.is_active = False
                    worker_last_completion_time[worker_id] = time.time()
                    if pid in pid_to_worker_id: del pid_to_worker_id[pid]
        except (EOFError, BrokenPipeError):
            break
        except Exception as e:
            log.error(f"Erreur dans le progress_listener: {e}", exc_info=True)


def report_collector(report_queue: multiprocessing.Queue, reports_list: List[Dict]):
    while True:
        try:
            report = report_queue.get()
            if report == "STOP": break
            reports_list.append(report)
        except (EOFError, BrokenPipeError):
            break

def display_final_report(reports_list: List[Dict]):
    if not reports_list:
        console.print("\n[bold green]✓ Rapport final : Aucune erreur ou avertissement à signaler.[/bold green]")
        return
    console.print("\n\n--- [bold red]Rapport Final des Problèmes Rencontrés[/bold red] ---")
    grouped_reports = defaultdict(list)
    for report in reports_list:
        grouped_reports[report['task_name']].append(report)
    for task_name, reports in sorted(grouped_reports.items()):
        panel_color = "red" if any(r['level'] == 'ERROR' for r in reports) else "yellow"
        table = Table(title=f"Tâche: [bold]{task_name}[/bold]", border_style=panel_color, show_header=True, header_style=f"bold {panel_color}")
        table.add_column("Niveau", style="dim", width=12)
        table.add_column("Message")
        for report in sorted(reports, key=lambda x: x['level']):
            level_style = "bold red" if report['level'] == 'ERROR' else "yellow"
            table.add_row(f"[{level_style}]{report['level']}[/{level_style}]", report['message'])
        console.print(table)

def generate_layout(overall_progress, num_workers) -> Layout:
    layout = Layout(name="root")
    layout.split(
        Layout(Panel(overall_progress), name="header", size=3),
        Layout(name="main_workers"),
    )
    
    panel_width_estimate = 48  
    num_cols = max(1, console.width // panel_width_estimate)

    rows = [Layout(name=f"row{i}") for i in range((num_workers + num_cols - 1) // num_cols)]
    with state_lock:
        worker_panels = [worker_states[i].get_renderable() for i in range(num_workers)]
    
    for i, row in enumerate(rows):
        row.split_row(*worker_panels[i*num_cols:(i+1)*num_cols])
    
    layout["main_workers"].split(*rows)
    return layout

def run_pool_in_thread(tasks, overall_progress, completion_event, num_procs):
    if num_procs == 0:
        completion_event.set()
        return
    with multiprocessing.Pool(processes=num_procs, maxtasksperchild=1) as pool:
        for _ in pool.imap_unordered(process_single_task, tasks):
            if overall_progress.tasks:
                overall_progress.update(overall_progress.tasks[0].id, advance=1)
    completion_event.set()

def process_single_task(task_args_tuple):
    (zone_id, zone_data, config, cycle_num, pass_id_num, ui_queue, report_queue) = task_args_tuple
    task_name = f"{zone_id}, Pass {str(pass_id_num).zfill(3)}, Cycle {str(cycle_num).zfill(3)}"
    pid = os.getpid()

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    stdout_relay = QueueRelay(ui_queue, pid, "log")
    stderr_relay = QueueRelay(ui_queue, pid, "log")
    
    sys.stdout = stdout_relay
    sys.stderr = stderr_relay

    logging.basicConfig(level=logging.INFO, stream=stderr_relay, force=True,
                        format='%(levelname)s: %(message)s')
    log_worker = logging.getLogger("worker_log")

    try:
        netcdf_output_path = analysis.process_and_save_zone_data(
            zone_id, zone_data, config, str(cycle_num).zfill(3), str(pass_id_num).zfill(3),
            ui_queue, report_queue, task_name)
        if netcdf_output_path:
            plotting.create_plots_from_netcdf(
                netcdf_filepath=netcdf_output_path, config=config,
                ui_queue=ui_queue, report_queue=report_queue, task_name=task_name)
    except Exception as e:
        tb_str = traceback.format_exc()
        log_worker.error(f"ERREUR CRITIQUE NON GÉRÉE DANS LA TÂCHE {task_name} (PID: {pid})")
        log_worker.error(tb_str)
        detailed_message = f"Erreur critique non gérée : {e}\n\nTraceback:\n{tb_str}"
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': detailed_message})
        ui_queue.put((pid, "final", f"[bold red]✗ Échec critique: {task_name}[/bold red]"))
    finally:
        stdout_relay.flush()
        stderr_relay.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    
    return True

def generate_tasks_from_filesystem(data_path: str, all_zones_info: Dict[str, Any], global_cfg: Dict[str, Any]) -> List[Tuple]:
    log.info(f"Génération des tâches à partir des fichiers dans : {data_path}")
    
    # Récupérer le type de données configuré
    data_type = global_cfg.get("data_type", "LR")  # Par défaut LR
    log.info(f"Type de données configuré : {data_type}")
    
    if not os.path.isdir(data_path):
        log.error(f"Le répertoire de données de base n'existe pas : {data_path}")
        return []
    
    tasks_to_process, processed_combinations = [], set()
    
    if data_type == "HR":
        search_path = os.path.join(data_path, "L2_HR")
        log.info(f"Recherche de fichiers HR dans : {search_path}")
    elif data_type == "LR":
        search_path = os.path.join(data_path, "L2_LR")
        log.info(f"Recherche de fichiers LR dans : {search_path}")
        if not os.path.exists(search_path):
            search_path = data_path
            log.info(f"Dossier L2_LR non trouvé, utilisation du dossier racine : {search_path}")
    else:
        log.error(f"Type de données non supporté : {data_type}. Utilisez 'HR' ou 'LR'.")
        return []
    
    if not os.path.isdir(search_path):
        log.error(f"Le répertoire de données {data_type} n'existe pas : {search_path}")
        return []
            
    for filename in os.listdir(search_path):
        if not filename.endswith(".nc"): 
            continue
            
        cycle_num, pass_id_num = None, None
        
        try:
            if data_type == "HR":
                # Format HR: SWOT_L2_HR_Raster_100m_UTM30T_N_x_x_x_{cycle}_{pass}_{tile}_...
                parts = filename.split('_')
                if len(parts) >= 13 and parts[0] == "SWOT" and parts[1] == "L2" and parts[2] == "HR":
                    cycle_num = int(parts[10])
                    pass_id_num = int(parts[11])
                    tile = parts[12]
                    log.debug(f"Fichier HR trouvé: cycle={cycle_num}, pass={pass_id_num}, tuile={tile}")
            elif data_type == "LR":
                # Format LR: SWOT_L2_LR_SSH_*_{cycle}_{pass}_... ou ancien format
                parts = filename.split('_')
                if len(parts) >= 7:
                    if parts[1] == "L2" and parts[2] == "LR":
                        cycle_num = int(parts[5])
                        pass_id_num = int(parts[6])
                    else:
                        cycle_num = int(parts[5])
                        pass_id_num = int(parts[6])
                    log.debug(f"Fichier LR trouvé: cycle={cycle_num}, pass={pass_id_num}")
            
            if cycle_num is None or pass_id_num is None:
                continue
                
            # Cherche les zones correspondant à ce pass
            for zone_id, zone_data in all_zones_info.items():
                if pass_id_num in zone_data.get("pass_id", []):
                    task_key = (zone_id, cycle_num, pass_id_num)
                    if task_key in processed_combinations: 
                        continue
                    
                    current_config_for_zone = global_cfg.copy()
                    current_config_for_zone["target_zone_id"] = zone_id
                    
                    if "extent" not in zone_data or "lon" not in zone_data["extent"] or "lat" not in zone_data["extent"]: 
                        continue
                    current_config_for_zone["analysis_roi_bbox_dict"] = zone_data["extent"]
                    
                    specific_tide_path = zone_data.get("tide_gauge_filepath")
                    if not specific_tide_path: 
                        continue
                    current_config_for_zone["tide_gauge_filepath"] = os.path.join(app_config.PROJECT_ROOT, specific_tide_path)
                    
                    task_args = (zone_id, zone_data, current_config_for_zone, cycle_num, pass_id_num)
                    tasks_to_process.append(task_args)
                    processed_combinations.add(task_key)
                    
                    log.info(f"Tâche ajoutée: {zone_id} - Cycle {cycle_num}, Pass {pass_id_num} (type: {data_type})")
                    
        except (ValueError, IndexError) as e:
            log.debug(f"Fichier au format non reconnu ignoré ({data_type}): {filename} - Erreur: {e}")
            continue
    
    log.info(f"Total des tâches générées: {len(tasks_to_process)}")
    return tasks_to_process

def main():
    main_handler = RichHandler(console=console, show_path=False, markup=True, log_time_format="[%X]")
    log.addHandler(main_handler)

    log.info("--- Début de l'Analyse Combinée SWOT et Ligne d'Eau ---")
    
    try:
        global_cfg = app_config.GLOBAL_CONFIG
        all_zones_info = app_config.load_zone_configurations(global_cfg["zone_info_filepath"])
    except FileNotFoundError as e:
        log.error(f"Erreur de configuration: {e}"); sys.exit(1)
    
    if not all_zones_info:
        log.warning("Aucune zone définie. Arrêt."); sys.exit(1)

    tasks_to_process = generate_tasks_from_filesystem(
        data_path=global_cfg["data_path"], all_zones_info=all_zones_info, global_cfg=global_cfg)
    
    if not tasks_to_process:
        log.info("Aucune tâche valide à traiter. Arrêt."); sys.exit(0)

    num_processes_config = global_cfg.get("num_processes")
    if num_processes_config and isinstance(num_processes_config, int) and num_processes_config > 0:
        num_processes = min(num_processes_config, os.cpu_count() or num_processes_config)
    else:
        num_processes = min(os.cpu_count() or 4, 8)
    
    log.info(f"\n[bold green]Préparation de {len(tasks_to_process)} tâches VALIDES à exécuter sur {num_processes} processus.[/bold green]")
    
    for i in range(num_processes): worker_states[i] = WorkerState()
    
    overall_progress = Progress(TextColumn("[bold]Progression totale...[/bold]"), BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", TextColumn("({task.completed} de {task.total})"))
    if tasks_to_process:
        overall_progress.add_task("Tâches", total=len(tasks_to_process))

    log.removeHandler(main_handler)

    final_reports = []

    with multiprocessing.Manager() as manager:
        ui_queue = manager.Queue()
        report_queue = manager.Queue()
        reports_list = manager.list()
        final_task_args = [task + (ui_queue, report_queue) for task in tasks_to_process]
        
        listener_thread = threading.Thread(target=progress_listener, args=(ui_queue, num_processes))
        report_thread = threading.Thread(target=report_collector, args=(report_queue, reports_list))
        pool_finished_event = threading.Event()
        pool_thread = threading.Thread(target=run_pool_in_thread, args=(final_task_args, overall_progress, pool_finished_event, num_processes))
        
        with Live(generate_layout(overall_progress, num_processes), screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
            listener_thread.start()
            report_thread.start()
            pool_thread.start()
            while not pool_finished_event.is_set():
                live.update(generate_layout(overall_progress, num_processes))
                time.sleep(0.1)
            pool_thread.join()
            time.sleep(0.5)
            live.update(generate_layout(overall_progress, num_processes))
            ui_queue.put("STOP")
            report_queue.put("STOP")
            listener_thread.join()
            report_thread.join()
        
        final_reports = list(reports_list)
    
    log.addHandler(main_handler)
    log.info("--- Fin de toutes les analyses ---")
    display_final_report(final_reports)

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()