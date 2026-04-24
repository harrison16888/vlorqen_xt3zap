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
    creds = None
    token_path = os.path.join(project_root, 'token.pickle')
    creds_path = os.path.join(project_root, 'credentials.json')

    # 1. Try Loading Token from Environment Variable First (For GitHub Actions)
    env_token = os.environ.get('GOOGLE_DRIVE_TOKEN')
    if env_token:
        try:
            from google.oauth2.credentials import Credentials
            token_dict = json.loads(env_token)
            creds = Credentials.from_authorized_user_info(token_dict, SCOPES)
        except Exception as e:
            print(f"Error parsing GOOGLE_DRIVE_TOKEN: {e}")

    # 2. Try Loading Token from File
    if not creds and os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            try:
                creds = pickle.load(token)
            except Exception:
                print("Error loading token.pickle, re-authenticating.")
                creds = None
    
    # Check if we have valid credentials
    if not creds or not creds.valid:
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
        # 3. Try Loading Client Secrets from Environment Variable if file is missing
        env_creds = os.environ.get('GOOGLE_DRIVE_CREDENTIALS')
        if not os.path.exists(creds_path) and env_creds:
            print("Writing credentials.json from GOOGLE_DRIVE_CREDENTIALS env variable...")
            with open(creds_path, 'w', encoding='utf-8') as f:
                f.write(env_creds)

        if not os.path.exists(creds_path):
            print(f"Error: {creds_path} not found.")
            print("Please provision credentials.json from Google Cloud Console (OAuth 2.0 Desktop Client).")
            return None
        
        flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
        creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    return creds

def get_calendar_service(project_root):
    creds = get_credentials(project_root)
    if not creds: return None
    return build('calendar', 'v3', credentials=creds)

def get_drive_service(project_root):
    creds = get_credentials(project_root)
    if not creds: return None
    return build('drive', 'v3', credentials=creds)

def build_daily_report_md(run_stats_path, target_date_str):
    if not os.path.exists(run_stats_path):
        print(f"Error: Stats file {run_stats_path} not found.")
        return None, None

    try:
        with open(run_stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
    except Exception as e:
        print(f"Error parsing stats JSON: {e}")
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
    summary, content = build_daily_report_md(run_stats_path, date_str)
    
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
