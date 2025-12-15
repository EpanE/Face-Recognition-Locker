{
  "places_to_modify": [
    {
      "symbol": "__init__",
      "lines_hint": "top of class Face_Register (constructor)",
      "reason": "add internal state variables for capture mode, auto settings, timing and movement tracking"
    },
    {
      "symbol": "GUI_info",
      "lines_hint": "function that builds the right-hand GUI (currently creates Step 1/2/3 widgets)",
      "reason": "add new widgets (mode selector, number-of-pics control, Start/Stop Auto button) and adjust Save button placement"
    },
    {
      "symbol": "GUI_get_input_name",
      "lines_hint": "handler for Input button",
      "reason": "ensure auto-state and counters reset/validated when a new face folder is created"
    },
    {
      "symbol": "create_face_folder",
      "lines_hint": "folder creation logic",
      "reason": "ensure ss_cnt reset and any auto counters/state reset when new folder is created"
    },
    {
      "symbol": "save_current_face",
      "lines_hint": "saves a single face image",
      "reason": "make it return success/failure (bool) or update a flag so auto flow can detect successful saves and increment counters"
    },
    {
      "symbol": "process",
      "lines_hint": "main loop that detects faces and updates frames; ends with win.after(20, self.process)",
      "reason": "implement auto-capture decision logic here (use after() scheduling), call save_current_face when capture conditions are met and stop auto when target reached"
    },
    {
      "symbol": "run",
      "lines_hint": "initial run sequence",
      "reason": "no major edits expected but ensure new GUI widgets are initialized before process(); verify order"
    }
  ],
  "gui_changes": [
    {
      "widget_type": "Label",
      "variable_name": "lbl_mode",
      "text": "Mode:",
      "grid_row": 10,
      "grid_col": 0,
      "notes": "small label placed on same row as save/start controls"
    },
    {
      "widget_type": "Radiobutton",
      "variable_name": "rb_manual",
      "text": "Manual",
      "grid_row": 10,
      "grid_col": 1,
      "notes": "bound to `capture_mode_var` (value 'manual'); default selected"
    },
    {
      "widget_type": "Radiobutton",
      "variable_name": "rb_auto",
      "text": "Auto",
      "grid_row": 10,
      "grid_col": 2,
      "notes": "bound to `capture_mode_var` (value 'auto')"
    },
    {
      "widget_type": "Label",
      "variable_name": "lbl_num_pics",
      "text": "Num pics:",
      "grid_row": 11,
      "grid_col": 0,
      "notes": "label for number-of-pictures Spinbox"
    },
    {
      "widget_type": "Spinbox",
      "variable_name": "input_num_pics",
      "text": "",
      "grid_row": 11,
      "grid_col": 1,
      "notes": "allows integer input; default `5`; min=1; stores desired number of captures for auto mode"
    },
    {
      "widget_type": "Button",
      "variable_name": "btn_start_auto",
      "text": "Start Auto",
      "grid_row": 11,
      "grid_col": 2,
      "notes": "starts auto capture; disabled when `capture_mode_var` != 'auto' or no face folder; toggles to 'Stop Auto' when running"
    },
    {
      "widget_type": "Button",
      "variable_name": "btn_save_manual",
      "text": "Save current face",
      "grid_row": 10,
      "grid_col": 3,
      "notes": "existing Save button kept for manual mode but re-gridded to not span 3 cols; bound to `save_current_face`"
    },
    {
      "widget_type": "Label",
      "variable_name": "lbl_auto_status",
      "text": "",
      "grid_row": 12,
      "grid_col": 0,
      "notes": "small feedback line for auto progress (e.g. 'Auto: 2/5 saved') - update `log_all` may be reused instead"
    }
  ],
  "state_vars": [
    {
      "name": "capture_mode",
      "initial_value": "'manual'",
      "purpose": "string 'manual' or 'auto' representing selected capture mode; mirrored by a Tkinter variable for GUI"
    },
    {
      "name": "capture_mode_var",
      "initial_value": "tk.StringVar(value='manual')",
      "purpose": "Tkinter variable bound to Radiobuttons to reflect GUI selection"
    },
    {
      "name": "auto_target_count",
      "initial_value": "5",
      "purpose": "how many pictures to take in auto mode (read from `input_num_pics` Spinbox)"
    },
    {
      "name": "auto_interval",
      "initial_value": "1.0",
      "purpose": "minimum seconds between automatic saves (default 1.0s)"
    },
    {
      "name": "auto_running",
      "initial_value": "False",
      "purpose": "bool flag indicating auto capture is active"
    },
    {
      "name": "last_capture_time",
      "initial_value": "0.0",
      "purpose": "timestamp (time.time()) of last auto capture; used to enforce `auto_interval` cooldown"
    },
    {
      "name": "last_bbox_center",
      "initial_value": "None",
      "purpose": "tuple (cx,cy) of last captured face bbox center; used to detect movement and avoid duplicate captures"
    },
    {
      "name": "require_movement_px",
      "initial_value": "20",
      "purpose": "pixel threshold for center movement required to allow another capture while face remains present"
    }
  ],
  "auto_capture_spec": {
    "default_interval_seconds": 1.0,
    "default_trigger_condition": "current_frame_faces_cnt == 1 AND out_of_range_flag == False AND face_folder_created_flag == True",
    "additional_checks": "auto_running == True; time since last_capture >= auto_interval; and either bbox center moved >= require_movement_px OR a face-absence-reset occurred (face disappeared then reappeared)",
    "stopping_conditions": [
      "ss_cnt (existing screenshot counter) >= auto_target_count -> stop auto",
      "user toggles 'Stop Auto' button (btn_start_auto) -> stop auto",
      "camera/frame error or face_folder_created_flag becomes False -> stop auto"
    ],
    "dedup_strategy": "combine cooldown + movement threshold: require at least `auto_interval` seconds since last capture AND that the current bbox center moved by at least `require_movement_px` pixels from `last_bbox_center`. Also set a fallback 'presence-reset' boolean: when len(faces)==0 set `presence_reset=True`; on next valid face set `last_bbox_center=None` (allow capture immediately after cooldown).",
    "interaction_with_save_current_face": "auto logic calls the same `save_current_face()` method; `save_current_face` should indicate success (return True) so auto flow can update `last_capture_time`, `last_bbox_center`, and stop when target reached."
  },
  "sequence_manual": [
    "User types name into `input_name` and clicks 'Input' -> `GUI_get_input_name()` reads the entry, calls `create_face_folder()` which increments `existing_faces_cnt`, creates folder and resets `ss_cnt` to 0; set `face_folder_created_flag=True`.",
    "User positions face in front of camera. `process()` shows live feed and updates `current_frame_faces_cnt` and `out_of_range_flag`.",
    "User clicks `btn_save_manual` ('Save current face') -> `save_current_face()` runs: checks `face_folder_created_flag`, `current_frame_faces_cnt==1`, and not out_of_range; on success `ss_cnt` increments and image written; GUI `log_all` updated.",
    "Repeat Save clicks until desired images collected; implementer may optionally enable `input_num_pics` to be only informative in manual mode."
  ],
  "sequence_auto": [
    "User types name into `input_name` and clicks 'Input' -> `GUI_get_input_name()` -> `create_face_folder()` resets `ss_cnt`.",
    "User sets desired number in `input_num_pics` Spinbox; this value is stored into `auto_target_count` (validate integer >=1).",
    "User selects 'Auto' Radiobutton and clicks `btn_start_auto` -> `start_auto_capture()` sets `auto_running=True`, disables `input_name`/`Input`/`Start Auto` (or toggles button to 'Stop Auto'), resets `last_capture_time=0` and `last_bbox_center=None` and updates `lbl_auto_status` or `log_all`.",
    "`process()` loop runs continuously; when `auto_running` is True and trigger conditions met (see `auto_capture_spec`) it calls `save_current_face()`; if `save_current_face()` returns True then update `last_capture_time=time.time()`, set `last_bbox_center` to current bbox center, update `lbl_auto_status` and `log_all`.",
    "When `ss_cnt` >= `auto_target_count` stop auto by calling `stop_auto_capture()` which sets `auto_running=False`, enables UI controls and notifies user (e.g., 'Auto complete: 5/5 saved')."
  ],
  "function_edits": [
    {
      "function": "__init__",
      "edit_summary": "add new state variables (capture_mode_var, capture_mode, auto_target_count, auto_interval, auto_running, last_capture_time, last_bbox_center, require_movement_px) and create/initialize `input_num_pics` UI variable placeholder"
    },
    {
      "function": "GUI_info",
      "edit_summary": "add Radiobuttons for Manual/Auto, Spinbox for number of pictures, Start/Stop Auto button, re-grid existing Save button; wire GUI controls to new handlers/vars"
    },
    {
      "function": "GUI_get_input_name",
      "edit_summary": "validate input, call create_face_folder(), reset auto-related counters/state (e.g., auto_running=False, last_capture_time=0) when a new name/folder is created"
    },
    {
      "function": "create_face_folder",
      "edit_summary": "ensure `ss_cnt` reset (already done) and also reset `last_capture_time`/`last_bbox_center`/`auto_running` to safe defaults"
    },
    {
      "function": "save_current_face",
      "edit_summary": "optionally return boolean success and avoid GUI-blocking operations; ensure it can be safely invoked from auto logic and updates `ss_cnt` as existing behaviour"
    },
    {
      "function": "process",
      "edit_summary": "after detection and drawing, add auto-capture decision logic that calls `save_current_face()` when conditions met; update `last_capture_time`/`last_bbox_center` and stop auto when target reached; use existing `win.after()` loop (no threads)"
    },
    {
      "function": "run",
      "edit_summary": "ensure new GUI widgets are initialized and that `process()` is started after GUI setup"
    }
  ],
  "new_helpers": [
    {
      "name": "start_auto_capture",
      "purpose": "validate inputs, set `auto_running=True`, read `input_num_pics` into `auto_target_count`, reset timers/counters, update UI (button text to 'Stop Auto')"
    },
    {
      "name": "stop_auto_capture",
      "purpose": "set `auto_running=False`, update UI (button text), re-enable controls, and log final status"
    },
    {
      "name": "auto_capture_check",
      "purpose": "called from `process()` to evaluate auto trigger conditions (time, movement, face count, out_of_range) and call `save_current_face()` when appropriate"
    },
    {
      "name": "bbox_center",
      "purpose": "small utility that returns (cx, cy) center coordinates for a dlib rectangle `d`"
    },
    {
      "name": "validate_num_pics",
      "purpose": "read and validate integer from `input_num_pics` Spinbox, coerce to minimum 1 and update `auto_target_count`"
    }
  ],
  "implementation_steps": [
    {
      "step": 1,
      "description": "Add state variables in Face_Register.__init__: `capture_mode_var` (tk.StringVar), `capture_mode`, `auto_target_count`, `auto_interval`, `auto_running`, `last_capture_time`, `last_bbox_center`, `require_movement_px` (file: get_faces_from_camera_tkinter.py; symbol: Face_Register.__init__)"
    },
    {
      "step": 2,
      "description": "Modify `GUI_info` to add Radiobuttons bound to `capture_mode_var`, add `input_num_pics` Spinbox, add `btn_start_auto` and reposition `btn_save_manual`; wire `btn_start_auto` to `start_auto_capture` (file: get_faces_from_camera_tkinter.py; symbol: Face_Register.GUI_info)"
    },
    {
      "step": 3,
      "description": "Add helpers `start_auto_capture`, `stop_auto_capture`, `auto_capture_check`, and `bbox_center` to the class and implement minimal validation and UI updates (file: get_faces_from_camera_tkinter.py)"
    },
    {
      "step": 4,
      "description": "Make `save_current_face` return True on successful save and False otherwise; do not change its image-writing behavior (file: get_faces_from_camera_tkinter.py; symbol: Face_Register.save_current_face)"
    },
    {
      "step": 5,
      "description": "Extend `process()` to call `auto_capture_check()` each loop when `auto_running` is True; on successful save update `last_capture_time` and `last_bbox_center`, and when `ss_cnt >= auto_target_count` call `stop_auto_capture()` (file: get_faces_from_camera_tkinter.py; symbol: Face_Register.process)"
    },
    {
      "step": 6,
      "description": "Test interactively: Manual flow (Input -> position face -> Save); Auto flow (Input -> set Num pics -> Start Auto -> observe saves until target). Use `win.after()` loop; avoid threads."
    }
  ]
}
