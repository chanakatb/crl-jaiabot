import time
import requests
import json

def get_current_state(bot_id, API_ENDPOINT_STATUS, headers):
    resp = requests.get(url=API_ENDPOINT_STATUS, headers=headers)
    data = json.loads(resp.text)
    current_bot_state = data['bots'][str(bot_id)]['mission_state']
    return current_bot_state

def send_dive_command(bot_id, max_depth, depth_interval, hold_time, drift_time, API_ENDPOINT_RC, headers):
    """Send dive command to a single bot"""
    dive_command = {
        'bot_id': bot_id,
        'type': "REMOTE_CONTROL_TASK",
        'rc_task': {
            'type': "DIVE",
            'dive': {
                'maxDepth': max_depth,
                'depthInterval': depth_interval,
                'holdTime': hold_time
            },
            'surface_drift': {'driftTime': drift_time}
        }
    }
    resp_dive_command = requests.post(url=API_ENDPOINT_RC, json=dive_command, headers=headers)    
    return resp_dive_command

def set_diving_new(bot_ids, max_depths, depth_intervals, hold_times, drift_times, API_ENDPOINT_RC, API_ENDPOINT_STATUS,  API_ENDPOINT_ALL_STOP):
    """
    Sets diving parameters for multiple bots with individual retry handling at each stage.
    """
    # Start Stopwatch
    start_time_diving = time.time()

    headers = {'clientid': 'backseat-control', 'Content-Type': 'application/json; charset=utf-8'}
    active_bots = set(bot_ids)
    
    # Step 0: Current state of all robots
    print('\n=== Step 0: Current state of all robots ===\n ')
    for bot_id in list(active_bots):
        current_state = get_current_state(bot_id, API_ENDPOINT_STATUS, headers)
        print(f"Current state of Robot {bot_id}: {current_state}")
    
    # # Step 1: Stop all bots
    # print('\n=== Step 1: Stop all bots ===\n ')
    # all_stop_resp = requests.post(url=API_ENDPOINT_ALL_STOP, headers=headers)
    # print(f"The Stop Command pastebin URL is: {all_stop_resp.text}")
    # time.sleep(1)

    # Step 2: Send RC mode command
    if active_bots:
        print('\n=== Step 2: Send RC mode command ===\n ')
        for bot_id in active_bots:
            rc_command = {
                'bot_id': bot_id,
                'time': 0,
                'type': "MISSION_PLAN",
                'plan': {
                    'start': "START_IMMEDIATELY",
                    'movement': "REMOTE_CONTROL",
                    'recovery': {
                        'recover_at_final_goal': False,
                        'location': {"lat": 41.661725, "lon": -71.272264}
                    }
                }
            }
            resp_activation_command = requests.post(url=API_ENDPOINT_RC, json=rc_command, headers=headers)
            print(f"RC Mode activation command response of Robot {bot_id}: {resp_activation_command.text}")
    time.sleep(1)

    # Step 3: Send dive commands
    if active_bots:
        print("\n=== Step 3: Send dive commands ===\n")
        for bot_id, max_depth, depth_interval, hold_time, drift_time in zip(bot_ids, max_depths, depth_intervals, hold_times, drift_times):
            if bot_id in active_bots:
                resp_dive_command = send_dive_command(bot_id, max_depth, depth_interval, hold_time, drift_time, API_ENDPOINT_RC, headers)
                print(f"Dive command response of Robot {bot_id}: {resp_dive_command.text}")
    time.sleep(1)
    
    # Step 4: Identify inactive bots and retry RC mode activation up to 5 times
    inactive_bots = set()
    print("\n=== Step 4: Identify inactive bots and retry RC mode activation up to 10 times ===\n")
    for bot_id in list(active_bots):
        current_state = get_current_state(bot_id, API_ENDPOINT_STATUS, headers)
        print(f"Current state of Robot {bot_id}: {current_state}")
        if current_state  not in {"IN_MISSION__UNDERWAY__MOVEMENT__REMOTE_CONTROL__SURFACE_DRIFT"}:
            inactive_bots.add(bot_id)    
    print("Inactive RC Mode Robots: ",inactive_bots)
    
    for attempt in range(20):
        if not inactive_bots:
            break
        print(f"\nAttempt {attempt + 1}/20: Retrying activation for inactive bots {inactive_bots}")
        for bot_id in list(inactive_bots):
            rc_command = {
                'bot_id': bot_id,
                'time': 0,
                'type': "MISSION_PLAN",
                'plan': {
                    'start': "START_IMMEDIATELY",
                    'movement': "REMOTE_CONTROL",
                    'recovery': {
                        'recover_at_final_goal': False,
                        'location': {"lat": 41.661725, "lon": -71.272264}
                    }
                }
            }
            resp_activation_command = requests.post(url=API_ENDPOINT_RC, json=rc_command, headers=headers)
            time.sleep(1)
            current_state = get_current_state(bot_id, API_ENDPOINT_STATUS, headers)
            print(f"Current state of Robot {bot_id}: {current_state}")
            if current_state in {"IN_MISSION__UNDERWAY__MOVEMENT__REMOTE_CONTROL__SURFACE_DRIFT", 
                                 "IN_MISSION__UNDERWAY__TASK__DIVE__DIVE_PREP",
                                 "IN_MISSION__UNDERWAY__TASK__DIVE__POWERED_DESCENT",
                                 "IN_MISSION__UNDERWAY__TASK__DIVE__HOLD",
                                 "IN_MISSION__UNDERWAY__TASK__DIVE__REACQUIRE_GPS"}: 
                inactive_bots.remove(bot_id)
        print("Inactive RC Mode Robots: ",inactive_bots)
        time.sleep(0.5)

    # Step 5: Identify non-diving bots and retry dive initiation up to 5 times
    non_diving_bots = set()
    print("\n=== Step 5: Identify non-diving bots and retry dive initiation up to 10 times ===\n")
    for bot_id in list(active_bots):
        current_state = get_current_state(bot_id, API_ENDPOINT_STATUS, headers)
        print(f"Current state of Robot {bot_id}: {current_state}")
        if current_state not in {
            "IN_MISSION__UNDERWAY__TASK__DIVE__DIVE_PREP",
            "IN_MISSION__UNDERWAY__TASK__DIVE__POWERED_DESCENT",
            "IN_MISSION__UNDERWAY__TASK__DIVE__HOLD",
            "IN_MISSION__UNDERWAY__TASK__DIVE__REACQUIRE_GPS"
        }:
            non_diving_bots.add(bot_id)

    print("Non diving robots: ",non_diving_bots)
    for attempt in range(20):
        if not non_diving_bots:
            break
        print(f"\nAttempt {attempt + 1}/20: Retrying dive initiation for non-diving bots {non_diving_bots}")
        for bot_id in list(non_diving_bots):
            send_dive_command(bot_id, max_depths[bot_ids.index(bot_id)], depth_intervals[bot_ids.index(bot_id)], hold_times[bot_ids.index(bot_id)], drift_times[bot_ids.index(bot_id)], API_ENDPOINT_RC, headers)
            time.sleep(1)
            current_state = get_current_state(bot_id, API_ENDPOINT_STATUS, headers)
            print(f"Current state of Robot {bot_id}: {current_state}")
            if current_state in {
                "IN_MISSION__UNDERWAY__TASK__DIVE__DIVE_PREP",
                "IN_MISSION__UNDERWAY__TASK__DIVE__POWERED_DESCENT",
                "IN_MISSION__UNDERWAY__TASK__DIVE__HOLD",
                "IN_MISSION__UNDERWAY__TASK__DIVE__REACQUIRE_GPS"
            }:
                non_diving_bots.remove(bot_id)
        print("Non diving robots: ",non_diving_bots)
        time.sleep(0.5)
    
    # End Stopwatch
    end_time_diving = time.time()
    elapsed_time_diving = end_time_diving - start_time_diving
    print(f"\nTotal time for dive: {elapsed_time_diving:.2f} seconds")

    # Diving status of all robots
    print("|||||||||" * 2, "  Bots that successfully started diving: ", active_bots - non_diving_bots, " ", "|||||||||" * 2)

