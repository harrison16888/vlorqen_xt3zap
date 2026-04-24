import os
import sys
import datetime
import json
import argparse
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Force UTF-8 for console output
sys.stdout.reconfigure(encoding='utf-8')

# Required scopes: need both calendar events and drive file upload permissions
SCOPES = ['https://www.googleapis.com/auth/calendar.events', 'https://www.googleapis.com/auth/drive.file']

def get_credentials(project_root):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    
    creds = None
    token_pickle_path = os.path.join(project_root, 'token.pickle')
    token_json_path = os.path.join(project_root, 'token.json')
    creds_path = os.path.join(project_root, 'credentials.json')

    # 1. Try Loading from token.json or token.pickle on disk
    token_file = None
    if os.path.exists(token_json_path):
        token_file = token_json_path
    elif os.path.exists(token_pickle_path):
        token_file = token_pickle_path

    if token_file:
        try:
            with open(token_file, "rb") as f:
                content = f.read()

            if content.startswith(b"\x80"):
                # It's a pickle format
                pkl_creds = pickle.loads(content)
                raw_scopes = getattr(pkl_creds, "_scopes", getattr(pkl_creds, "scopes", None))
                granted_scopes = list(raw_scopes) if raw_scopes else SCOPES
                creds_dict = {
                    "token":         getattr(pkl_creds, "token", None),
                    "refresh_token": getattr(pkl_creds, "_refresh_token", getattr(pkl_creds, "refresh_token", None)),
                    "token_uri":     getattr(pkl_creds, "_token_uri", getattr(pkl_creds, "token_uri", "https://oauth2.googleapis.com/token")),
                    "client_id":     getattr(pkl_creds, "_client_id", getattr(pkl_creds, "client_id", None)),
                    "client_secret": getattr(pkl_creds, "_client_secret", getattr(pkl_creds, "client_secret", None)),
                    "scopes":        granted_scopes,
                }
                creds = Credentials.from_authorized_user_info(creds_dict, granted_scopes)
            else:
                # It's JSON format
                creds_dict = json.loads(content)
                granted_scopes = creds_dict.get("scopes", SCOPES)
                if isinstance(granted_scopes, str):
                    granted_scopes = granted_scopes.split()
                creds = Credentials.from_authorized_user_info(creds_dict, granted_scopes)
        except Exception as e:
            print(f"Error loading {token_file}: {e}")
            creds = None
            
    # Check if we have valid credentials
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as e:
            print(f"Error refreshing token: {e}")
            creds = None

    # Verify scopes are present
    if creds and creds.valid:
        if not all(scope in creds.scopes for scope in SCOPES):
            print(f"Missing required scopes. Current: {creds.scopes}. Re-authenticating...")
            creds = None

    if not creds:
        # Fallback to Interactive OAuth Flow
        # GitHub actions creates credentials.json from GOOGLE_DRIVE_CREDENTIALS but that might be malformed if done improperly.
        # However, in GA, we expect token.json to already exist (created by `pipeline.yml` string decode).
        if not os.path.exists(creds_path):
            print(f"Error: Neither token.json, token.pickle nor {creds_path} contains valid credentials.")
            return None
        
        try:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
            with open(token_pickle_path, 'wb') as token:
                pickle.dump(creds, token)
        except Exception as e:
            print(f"Failed to run InstalledAppFlow from {creds_path}: {e}")
            return None

    return creds

def get_calendar_service(project_root):
    creds = get_credentials(project_root)
    if not creds: return None
    return build('calendar', 'v3', credentials=creds)

def get_drive_service(project_root):
    creds = get_credentials(project_root)
    if not creds: return None
    return build('drive', 'v3', credentials=creds)

def fetch_run_stats_from_drive(service, target_date_str):
    import io
    from googleapiclient.http import MediaIoBaseDownload
    
    dt = datetime.datetime.strptime(target_date_str, "%Y-%m-%d")
    year_str = dt.strftime("%Y")
    month_str = dt.strftime("%m")
    
    root_id = "1tnTb4BjVjOARRKaQjmrse4kddddj9ogj"
    log_filename = f"pipeline_stats_{year_str}_{month_str}.jsonl"
    
    combined = {
        "run_ts": f"{target_date_str} 23:59",
        "llm_primary": "Unknown",
        "llm_total_calls": 0, "topics_fetched": 0, "topics_approved": 0,
        "topics_skipped": 0, "images_ok": 0, "images_failed": 0, "audio_ok": 0,
        "titles": [], "errors": []
    }
    
    try:
        q = f"name='{log_filename}' and '{root_id}' in parents and trashed=false"
        files = service.files().list(q=q, fields="files(id)").execute().get("files", [])
        if files:
            fh = io.BytesIO()
            dl = MediaIoBaseDownload(fh, service.files().get_media(fileId=files[0]["id"]))
            done = False
            while not done:
                _, done = dl.next_chunk()
            records = [json.loads(line) for line in fh.getvalue().decode("utf-8").splitlines() if line.strip()]
            day_records = [r for r in records if r.get("run_ts", "").startswith(target_date_str)]
            
            if day_records:
                combined = day_records[-1].copy()
                combined["llm_total_calls"] = sum(r.get("llm_total_calls", 0) for r in day_records)
                combined["topics_fetched"] = sum(r.get("topics_fetched", 0) for r in day_records)
                combined["topics_approved"] = sum(r.get("topics_approved", 0) for r in day_records)
                combined["topics_skipped"] = sum(r.get("topics_skipped", 0) for r in day_records)
                combined["images_ok"] = sum(r.get("images_ok", 0) for r in day_records)
                combined["images_failed"] = sum(r.get("images_failed", 0) for r in day_records)
                combined["audio_ok"] = sum(r.get("audio_ok", 0) for r in day_records)
                combined["titles"] = list(set(t for r in day_records for t in r.get("titles", [])))
                combined["errors"] = [err for r in day_records for err in r.get("errors", [])]
    except Exception as e:
        print(f"Error fetching stats log from Drive: {e}")

    # Explicitly fetch ALL subfolders from today's Drive folder for accuracy
    def find_folder(parent_id, name):
        try:
            q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
            files = service.files().list(q=q, fields="files(id)").execute().get("files", [])
            return files[0]["id"] if files else None
        except Exception: return None

    try:
        import re
        year_id = find_folder(root_id, year_str)
        if year_id:
            month_id = find_folder(year_id, month_str)
            if month_id:
                date_id = find_folder(month_id, target_date_str)
                if date_id:
                    folder_q = f"'{date_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                    topic_folders = service.files().list(q=folder_q, fields="files(name)").execute().get("files", [])
                    drive_titles = []
                    for f in topic_folders:
                        m = re.search(r"^\d{4}-(.+)", f["name"])
                        title = m.group(1).replace("-", " ") if m else f["name"]
                        drive_titles.append(title)
                    if drive_titles:
                        print(f"Found {len(drive_titles)} topics explicitly in Drive subfolders.")
                        combined["titles"] = drive_titles
                        if not combined.get("topics_approved"):
                            combined["topics_approved"] = len(drive_titles)
    except Exception as e:
        print(f"Error listing subfolders on Drive: {e}")

    if not combined["titles"] and combined["llm_total_calls"] == 0:
        return None  # Nothing happened today
        
    return combined

def build_daily_report_md(run_stats_path, target_date_str, drive_service=None):
    stats = None
    if os.path.exists(run_stats_path):
        try:
            with open(run_stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
        except Exception as e:
            print(f"Error parsing stats JSON: {e}")
            
    if not stats and drive_service:
        print(f"Local {run_stats_path} missing. Attempting to fetch day stats from Google Drive...")
        stats = fetch_run_stats_from_drive(drive_service, target_date_str)

    if not stats:
        print(f"Error: Stats for {target_date_str} not found locally or on Drive.")
        return None, None

    titles = stats.get('titles', [])
    summary = f"Public AI Daily Report: {target_date_str}"
    
    content_lines = [
        f"# {summary}",
        "",
        f"**Run Timestamp:** {stats.get('run_ts', 'Unknown')}",
        f"**Primary LLM Engine:** `{stats.get('llm_primary', 'Unknown')}`",
        f"**Total LLM Calls:** {stats.get('llm_total_calls', 0)}",
        "",
        "## Statistics",
        f"- Topics Fetched: {stats.get('topics_fetched', 0)}",
        f"- Topics Approved: {stats.get('topics_approved', 0)}",
        f"- Topics Skipped: {stats.get('topics_skipped', 0)}",
        f"- Images Successfully Generated: {stats.get('images_ok', 0)}",
        f"- Image Failures: {stats.get('images_failed', 0)}",
        f"- Audio Successes: {stats.get('audio_ok', 0)}",
        "",
        "## Top News Topics Handled Today"
    ]

    for title in titles:
        content_lines.append(f"- {title}")

    if stats.get('errors'):
        content_lines.append("")
        content_lines.append("## Errors Encountered")
        for err in stats.get('errors', []):
            content_lines.append(f"- {err}")

    content = "\n".join(content_lines)
    return summary, content

def upload_to_drive(service, file_path):
    file_metadata = {'name': os.path.basename(file_path)}
    
    ext = os.path.splitext(file_path)[1].lower()
    mimetype = 'application/octet-stream'
    if ext == '.md': mimetype = 'text/markdown'
    elif ext in ['.png', '.jpg', '.jpeg']:
        mimetype = f'image/{ext[1:]}'
        if ext == '.jpg': mimetype = 'image/jpeg'

    media = MediaFileUpload(file_path, mimetype=mimetype)
    try:
        file = service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
        print(f"File uploaded to Drive: {file.get('webViewLink')}")
        return file.get('id'), file.get('webViewLink')
    except Exception as e:
        print(f"Error uploading to Drive: {e}")
        return None, None

def add_all_day_event(service, date_str, summary, description, attachments=None):
    if attachments:
        description += "\n\n--- Attachments ---\n"
        for att in attachments:
            description += f"- {att['title']}: {att['fileUrl']}\n"

    event = {
        'summary': summary,
        'description': description,
        'start': {'date': date_str},
        'end': {'date': date_str},
    }

    if attachments:
        event['attachments'] = attachments

    # End date is exclusive in all-day events
    start_dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    end_dt = start_dt + datetime.timedelta(days=1)
    event['end']['date'] = end_dt.strftime('%Y-%m-%d')

    try:
        result = service.events().insert(calendarId='primary', body=event, supportsAttachments=True).execute()
        print(f"Event created: {result.get('htmlLink')}")
    except Exception as e:
        print(f"An error occurred while creating event: {e}")

def list_events(service, date_str):
    print(f"Listing events for {date_str} (and surrounding days)...")
    start_dt = datetime.datetime.strptime(date_str, '%Y-%m-%d') - datetime.timedelta(days=1)
    end_dt = start_dt + datetime.timedelta(days=3)
    
    time_min = start_dt.isoformat() + 'Z'
    time_max = end_dt.isoformat() + 'Z'
    
    events_result = service.events().list(calendarId='primary', timeMin=time_min, timeMax=time_max,
                                        singleEvents=True, orderBy='startTime').execute()
    events = events_result.get('items', [])

    if not events:
        print('No events found in range.')
        return

    print(f"{'Date':<12} | {'Summary':<50} | {'Attachments'}")
    print("-" * 90)
    for event in events:
        start = event['start'].get('date', event['start'].get('dateTime', ''))[:10]
        summary = event.get('summary', 'No Summary')[:50]
        has_attachments = 'Yes' if 'attachments' in event else 'No'
        print(f"{start:<12} | {summary:<50} | {has_attachments}")
        if 'attachments' in event:
            for att in event['attachments']:
                print(f"  - Attachment: {att['title']} ({att['fileUrl']})")

def delete_existing_events(service, date_str):
    print(f"Checking for existing reports on {date_str}...")
    start_dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    end_dt = start_dt + datetime.timedelta(days=1)
    
    time_min = start_dt.isoformat() + 'Z'
    time_max = end_dt.isoformat() + 'Z'
    
    events_result = service.events().list(calendarId='primary', timeMin=time_min, timeMax=time_max,
                                        singleEvents=True).execute()
    events = events_result.get('items', [])

    for event in events:
        summary = event.get('summary', '')
        event_start = event['start'].get('date', event['start'].get('dateTime', ''))
        
        # Look for the exact same event signature
        if event_start.startswith(date_str) and summary.startswith("Public AI Daily Report:"):
            print(f"Deleting existing event: {summary} (ID: {event['id']})")
            service.events().delete(calendarId='primary', eventId=event['id']).execute()

def main():
    parser = argparse.ArgumentParser(description='Send daily summary report to Google Calendar.')
    parser.add_argument('--dry-run', action='store_true', help='Parse the stats but do not add to calendar.')
    parser.add_argument('--list', type=str, help='List events for a specific date (YYYY-MM-DD).')
    parser.add_argument('--date', type=str, help='Sync a specific date (YYYY-MM-DD). Defaults to today.')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..'))

    if args.list:
        service = get_calendar_service(project_root)
        if not service: return
        list_events(service, args.list)
        return

    # Use specified date or today
    if args.date:
        try:
            target_date = datetime.datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            print("Error: Invalid date format. Use YYYY-MM-DD.")
            return
    else:
        target_date = datetime.date.today()

    date_str = target_date.strftime('%Y-%m-%d')
    base_dir = os.path.join(project_root, "news")
    run_stats_path = os.path.join(base_dir, "run_stats.json")

    print(f"Parsing {run_stats_path} for date {date_str}...")
    drive_service_for_stats = get_drive_service(project_root)
    summary, content = build_daily_report_md(run_stats_path, date_str, drive_service_for_stats)
    
    if not summary:
        print("Could not generate daily report from stats. Exiting.")
        return

    # Write markdown summary to disk
    md_filename = f"daily_summary_{date_str}.md"
    md_path = os.path.join(base_dir, md_filename)
    try:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Generated local Markdown report at {md_path}")
    except Exception as e:
        print(f"Warning: Failed to save markdown locally: {e}")

    print(f"Ready to add event: {summary} on {date_str}")
    
    if args.dry_run:
        print("Dry run: Skipping calendar API calls.")
        return

    calendar_service = get_calendar_service(project_root)
    if not calendar_service: return
    
    delete_existing_events(calendar_service, date_str)

    attachments = []
    drive_service = get_drive_service(project_root)

    if drive_service:
        print(f"Uploading markdown report: {md_path}")
        file_id, file_url = upload_to_drive(drive_service, md_path)
        if file_id:
            attachments.append({
                'fileId': file_id,
                'fileUrl': file_url,
                'title': md_filename,
                'mimeType': 'text/markdown'
            })
        
        # Here we could also search and upload today's covers or images, 
        # similar to The_Day_In_History if needed:
        # e.g., scanning `news/YYYY/MM/DD` directories for `.png` files.
    else:
        print("Warning: Could not get Drive service, skipping attachments.")

    print(f"Total attachments to add: {len(attachments)}")
    add_all_day_event(calendar_service, date_str, summary, content, attachments)

if __name__ == '__main__':
    main()
