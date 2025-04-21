import time
import requests
import json
from functools import wraps

def retry_on_timeout(max_retries=10, timeout=10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except TimeoutError as e:
                    print(f"\nTimeout occurred (Attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        print("Restarting function...")
                        time.sleep(1)  # Wait before retrying
                    else:
                        print("Max retries reached. Giving up.")
                        raise
            return None
        return wrapper
    return decorator

@retry_on_timeout(max_retries=10, timeout=10)
def set_diving(ID, max_depth, depth_interval, hold_time, drift_time, API_ENDPOINT_RC, API_ENDPOINT_STATUS, API_ENDPOINT_ALL_STOP):    
    print(f"Bot ID: {ID}  Max depth: {max_depth} Depth interval:  {depth_interval}  Hold time: {hold_time} Drift time: {drift_time}")
    
    headers = {'clientid': 'backseat-control', 'Content-Type' : 'application/json; charset=utf-8'}        
            
    def check_bot_expected_mission_state(expected_states):
        current_bot_state = None
        start_time = time.time()
        
        while current_bot_state not in expected_states:
            # Check if we've exceeded the timeout
            if time.time() - start_time > 5:  # 5 second timeout
                raise TimeoutError(f"Timeout waiting for bot {ID} to enter expected states: {expected_states}")
            
            resp = requests.get(url=API_ENDPOINT_STATUS, headers=headers)
            data = json.loads(resp.text)
            current_bot_state = data['bots'][str(ID)]['mission_state']
            print(f"\nWaiting for bot mission state to enter: {expected_states} \n Current bot {ID} mission state: {current_bot_state}")
            time.sleep(1)
            
        print(f"\nMission state for bot {ID}: {current_bot_state} == one of the expected states {expected_states}")
    
    # Rest of your function remains the same
    all_stop_resp = requests.post(url=API_ENDPOINT_ALL_STOP, headers=headers)
    print(f"Bot ID: {ID} -> The Stop Command pastebin URL is:%s"%all_stop_resp.text)

    print('='*50, f'\n Bot: {ID} - STOP CHECK \n','='*50)
    check_bot_expected_mission_state(["IN_MISSION__UNDERWAY__RECOVERY__STOPPED"])
    
    print('='*50, f'\n Bot: {ID} - ACTIVATE CHECK \n','='*50)
    check_bot_expected_mission_state([
        "PRE_DEPLOYMENT__IDLE", 
        "POST_DEPLOYMENT__IDLE", 
        "PRE_DEPLOYMENT__FAILED", 
        "IN_MISSION__UNDERWAY__MOVEMENT__REMOTE_CONTROL__SURFACE_DRIFT",
        "IN_MISSION__UNDERWAY__RECOVERY__STOPPED"
    ])
    
    print(' Activate Bots')
    activate_command = {'bot_id':ID,'type':"ACTIVATE"}
    resp_activate_command = requests.post(url=API_ENDPOINT_RC, json=activate_command, headers=headers)
    print(f"Bot ID: {ID} -> The Activate Command pastebin URL is:%s"%resp_activate_command.text)

    print('='*50, f'\n Bot: {ID} - RC MODE CHECK \n','='*50)
    check_bot_expected_mission_state([
        "PRE_DEPLOYMENT__WAIT_FOR_MISSION_PLAN", 
        "IN_MISSION__UNDERWAY__MOVEMENT__REMOTE_CONTROL__SURFACE_DRIFT", 
        "IN_MISSION__UNDERWAY__RECOVERY__STOPPED"
    ])
    
    print(' Activate RC mode')
    rc_command = {'bot_id':ID,'time':0,'type':"MISSION_PLAN",'plan':{
        'start':"START_IMMEDIATELY",
        'movement':"REMOTE_CONTROL",
        'recovery':{'recover_at_final_goal':False,'location':{"lat":41.661725,"lon":-71.272264}}
    }}
    resp_rc_command = requests.post(url=API_ENDPOINT_RC, json=rc_command, headers=headers)
    print("The RC Command pastebin URL is:%s"%resp_rc_command.text)                
    
    print('='*50, f'\n Bot: {ID} - SEND BOT COMMAND CHECK \n','='*50)
    check_bot_expected_mission_state(["IN_MISSION__UNDERWAY__MOVEMENT__REMOTE_CONTROL__SURFACE_DRIFT"])
    
    print(' Dive mode')
    dive_command = {'bot_id':ID,'type':"REMOTE_CONTROL_TASK",'rc_task':{
        'type':"DIVE",
        'dive':{'maxDepth':max_depth,'depthInterval':depth_interval,'holdTime':hold_time},
        'surface_drift':{'driftTime':drift_time}
    }}
    resp_dive_command = requests.post(url=API_ENDPOINT_RC, json=dive_command, headers=headers)
    print("The Dive Command pastebin URL is:%s"%resp_dive_command.text)  

    print('='*50, f'\n Bot: {ID} - DIVNING START CHECK \n','='*50)
    check_bot_expected_mission_state([
        "IN_MISSION__UNDERWAY__TASK__DIVE__DIVE_PREP", 
        "IN_MISSION__UNDERWAY__TASK__DIVE__POWERED_DESCENT", 
        "IN_MISSION__UNDERWAY__TASK__DIVE__REACQUIRE_GPS",
        "IN_MISSION__UNDERWAY__MOVEMENT__REMOTE_CONTROL__SURFACE_DRIFT" # New one
    ])
    
    print('\n', '=-'*25, f'Bot {ID} started diving', '=-'*25,'\n')
    time.sleep(1)
