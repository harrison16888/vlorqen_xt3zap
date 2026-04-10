import os
import json
import sys
import pickle
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Root folder on Drive where news/YYYY/MM/... will be mirrored
DRIVE_ROOT_FOLDER_ID = '1tnTb4BjVjOARRKaQjmrse4kddddj9ogj'

FALLBACK_SCOPES = [
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive.readonly',
]

def get_drive_service():
    token_paths = ['token.json', os.path.expanduser('~/.api_tools/token.json')]
    token_file = next((p for p in token_paths if os.path.exists(p)), None)

    if not token_file:
        print("❌ Missing token.json.")
        return None

    with open(token_file, 'rb') as f:
        raw = f.read()

    if raw.startswith(b'\x80'):
        # Pickle format — convert to JSON
        pkl_creds = pickle.loads(raw)
        raw_scopes = getattr(pkl_creds, '_scopes', getattr(pkl_creds, 'scopes', None))
        granted_scopes = list(raw_scopes) if raw_scopes else FALLBACK_SCOPES
        creds_dict = {
            "token":         getattr(pkl_creds, 'token', None),
            "refresh_token": getattr(pkl_creds, '_refresh_token', getattr(pkl_creds, 'refresh_token', None)),
            "token_uri":     getattr(pkl_creds, '_token_uri', getattr(pkl_creds, 'token_uri', 'https://oauth2.googleapis.com/token')),
            "client_id":     getattr(pkl_creds, '_client_id', getattr(pkl_creds, 'client_id', None)),
            "client_secret": getattr(pkl_creds, '_client_secret', getattr(pkl_creds, 'client_secret', None)),
            "scopes":        granted_scopes,
        }
        with open(token_file, 'w') as f:
            json.dump(creds_dict, f, indent=2)
    else:
        creds_dict = json.loads(raw)
        granted_scopes = creds_dict.get('scopes', FALLBACK_SCOPES)
        if isinstance(granted_scopes, str):
            granted_scopes = granted_scopes.split()

    creds = Credentials.from_authorized_user_info(creds_dict, granted_scopes)

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(token_file, 'w') as f:
            f.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)


def find_existing_folder(service, name, parent_id):
    query = (
        f"name = '{name}' "
        f"and mimeType = 'application/vnd.google-apps.folder' "
        f"and '{parent_id}' in parents "
        f"and trashed = false"
    )
    results = service.files().list(q=query, fields='files(id, name)').execute()
    files = results.get('files', [])
    if files:
        print(f"📂 Found existing folder '{name}' (ID: {files[0]['id']})")
        return files[0]['id']
    return None


def get_or_create_folder(service, name, parent_id):
    existing_id = find_existing_folder(service, name, parent_id)
    if existing_id:
        return existing_id
    print(f"📁 Creating folder '{name}'...")
    meta = {'name': name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent_id]}
    folder = service.files().create(body=meta, fields='id').execute()
    fid = folder.get('id')
    print(f"✅ Created (ID: {fid})")
    return fid


def upload_file(service, local_path, parent_id):
    file_name = os.path.basename(local_path)
    print(f"📤 Uploading {file_name}...")
    meta = {'name': file_name, 'parents': [parent_id]}
    media = MediaFileUpload(local_path, resumable=True)
    f = service.files().create(body=meta, media_body=media, fields='id').execute()
    print(f"✅ Done (ID: {f.get('id')})")
    return f.get('id')


def upload_directory(service, local_dir, parent_id):
    """Recursively upload local_dir into parent_id, reusing existing folders."""
    dir_name = os.path.basename(local_dir.rstrip('/\\'))
    folder_id = get_or_create_folder(service, dir_name, parent_id)
    for item in sorted(os.listdir(local_dir)):
        item_path = os.path.join(local_dir, item)
        if os.path.isfile(item_path):
            upload_file(service, item_path, folder_id)
        elif os.path.isdir(item_path):
            upload_directory(service, item_path, folder_id)
    return folder_id


def resolve_drive_parent(service, local_path, local_base, drive_root_id):
    """
    Given a local path like /runner/.../news/2026/04/2026-04-09-News-...,
    and local_base like /runner/.../news,
    ensure the YYYY/MM path exists on Drive under drive_root_id,
    and return the Drive folder ID for the immediate parent of the upload target.

    If local_path IS the base (i.e. uploading the whole news tree), return drive_root_id.
    """
    rel = os.path.relpath(local_path, local_base)  # e.g. "2026/04/2026-04-09-..."
    parts = rel.split(os.sep)                       # ['2026', '04', '2026-04-09-...']

    if parts == ['.'] or parts == ['']:
        # Uploading from base itself — no intermediate folders needed
        return drive_root_id

    # Walk all parts except the last one (the item itself) to build the parent chain
    current_id = drive_root_id
    for part in parts[:-1]:
        current_id = get_or_create_folder(service, part, current_id)
    return current_id


def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_to_drive.py <local_file_or_directory>")
        sys.exit(1)

    local_path = os.path.abspath(sys.argv[1])
    if not os.path.exists(local_path):
        print(f"❌ Error: {local_path} does not exist.")
        sys.exit(1)

    service = get_drive_service()
    if not service:
        return

    # Determine the local news root so we can mirror YYYY/MM on Drive
    # Assumption: script lives in scripts/, news root is ../news/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_news_root = os.path.normpath(os.path.join(script_dir, '..', 'news'))

    # If the path being uploaded is inside the news root, mirror the structure.
    # Otherwise just upload directly under the Drive root folder.
    try:
        rel = os.path.relpath(local_path, local_news_root)
        is_under_news = not rel.startswith('..')
    except ValueError:
        is_under_news = False

    if is_under_news:
        drive_parent_id = resolve_drive_parent(service, local_path, local_news_root, DRIVE_ROOT_FOLDER_ID)
    else:
        drive_parent_id = DRIVE_ROOT_FOLDER_ID

    if os.path.isfile(local_path):
        upload_file(service, local_path, drive_parent_id)
    elif os.path.isdir(local_path):
        upload_directory(service, local_path, drive_parent_id)
    else:
        print("❌ Unsupported file type.")


if __name__ == "__main__":
    main()