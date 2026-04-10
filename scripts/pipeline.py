import os
import sys
import xml.etree.ElementTree as ET
import urllib.request
import json
import time
import subprocess
from datetime import datetime, timedelta, timezone
import re
import base64
import io
import random
import pickle
from PIL import Image

API_KEY = os.environ.get("API_KEY", "password")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8045/v1")
MAX_RETRIES = 3
TOPIC_LIMIT = int(os.environ.get("TOPIC_LIMIT", "3"))

# Percentage (0.0 - 1.0) chance that a given story uses artistic mode vs photographic mode
ART_PERCENTAGE = float(os.environ.get("ART_PERCENTAGE", "0.33"))

# Percentage (0.0 - 1.0) chance that human subjects are replaced with anthropomorphic animals
ANTHROPOMORPHIC_PERCENTAGE = float(os.environ.get("ANTHROPOMORPHIC_PERCENTAGE", "0.3"))

# Popular artistic styles cycled IN ORDER (round-robin). Never randomized.
# Set via env as a pipe-separated string, e.g.:
#   POPULAR_ARTISTIC_STYLES="Studio Ghibli watercolor|Pixar 3D cinematic|Synthwave retro 80s neon"
_ARTISTIC_STYLES_DEFAULT = (
    "Studio Ghibli watercolor illustration style|"
    "Pixar 3D cinematic render style|"
    "Synthwave retro 80s neon poster style|"
    "Ukiyo-e Japanese woodblock print style|"
    "Marvel Comics bold ink and color style|"
    "LEGO brick diorama style|"
    "Impressionist oil painting style (Monet)|"
    "GTA V loading screen poster style|"
    "Cyberpunk 2077 concept art style|"
    "Disney golden-age hand-drawn animation style"
)
POPULAR_ARTISTIC_STYLES = [
    s.strip()
    for s in os.environ.get("POPULAR_ARTISTIC_STYLES", _ARTISTIC_STYLES_DEFAULT).split("|")
    if s.strip()
]

# Persistent index tracking which style to use next (cycles through the list)
_artistic_style_index = 0

def get_drive_service():
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        print("Google API clients not installed. Please pip install google-api-python-client google-auth-oauthlib")
        return None

    FALLBACK_SCOPES = [
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/drive.readonly',
    ]
    creds = None

    token_paths = ['token.json', os.path.expanduser('~/.api_tools/token.json')]
    token_file = None
    for path in token_paths:
        if os.path.exists(path):
            token_file = path
            break

    if token_file:
        try:
            with open(token_file, 'rb') as f:
                content = f.read()

            if content.startswith(b'\x80'):
                print(f"🔄 Detected Pickle token format for {token_file} — converting to JSON...")
                pkl_creds = pickle.loads(content)

                raw_scopes = getattr(pkl_creds, '_scopes',
                             getattr(pkl_creds, 'scopes', None))
                granted_scopes = list(raw_scopes) if raw_scopes else FALLBACK_SCOPES

                creds_dict = {
                    "token":         getattr(pkl_creds, 'token', None),
                    "refresh_token": getattr(pkl_creds, '_refresh_token',
                                     getattr(pkl_creds, 'refresh_token', None)),
                    "token_uri":     getattr(pkl_creds, '_token_uri',
                                     getattr(pkl_creds, 'token_uri',
                                     'https://oauth2.googleapis.com/token')),
                    "client_id":     getattr(pkl_creds, '_client_id',
                                     getattr(pkl_creds, 'client_id', None)),
                    "client_secret": getattr(pkl_creds, '_client_secret',
                                     getattr(pkl_creds, 'client_secret', None)),
                    "scopes":        granted_scopes,
                }

                with open(token_file, 'w', encoding='utf-8') as f:
                    json.dump(creds_dict, f, indent=2)
                print(f"✅ Pickle converted and saved as JSON to {token_file}")
                print(f"   Granted scopes: {granted_scopes}")

                creds = Credentials.from_authorized_user_info(creds_dict, granted_scopes)

            else:
                print(f"📄 Detected JSON token format for {token_file}")
                creds_dict = json.loads(content)
                granted_scopes = creds_dict.get('scopes', FALLBACK_SCOPES)
                if isinstance(granted_scopes, str):
                    granted_scopes = granted_scopes.split()
                creds = Credentials.from_authorized_user_info(creds_dict, granted_scopes)

            if creds and creds.expired and creds.refresh_token:
                print("♻️ Token expired, refreshing...")
                creds.refresh(Request())
                with open(token_file, 'w', encoding='utf-8') as f:
                    f.write(creds.to_json())
                print("✅ Refreshed token saved.")

        except Exception as e:
            print(f"❌ Auth error reading {token_file}: {e}")

    if not creds or not creds.valid:
        print("⚠️ Google Drive API not authenticated properly. Missing or invalid token. Cannot sync opinions.")
        return None

    return build('drive', 'v3', credentials=creds)

def get_pacific_time():
    """Returns the current time in Pacific Time (PT)."""
    # Offset is -7 hours during PDT (current metadata shows -07:00)
    return datetime.now(timezone(timedelta(hours=-7)))

def get_todays_processed_titles(service):
    """Browses Drive to find folders created today in PT and returns their titles."""
    pt_now = get_pacific_time()
    year = pt_now.strftime("%Y")
    month = pt_now.strftime("%m")
    date = pt_now.strftime("%Y-%m-%d")
    
    root_id = '1tnTb4BjVjOARRKaQjmrse4kddddj9ogj'
    print(f"🔍 Checking Drive history for PT date: {date}")
    
    def find_folder(parent_id, name):
        try:
            query = f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed = false"
            results = service.files().list(q=query, fields='files(id, name)').execute()
            files = results.get('files', [])
            return files[0]['id'] if files else None
        except Exception as e:
            print(f"   ⚠️ Error finding folder '{name}': {e}")
            return None

    # Navigate: Root -> Year -> Month -> Date
    year_id = find_folder(root_id, year)
    if not year_id: return []
    month_id = find_folder(year_id, month)
    if not month_id: return []
    date_id = find_folder(month_id, date)
    if not date_id:
        print(f"   ℹ️ No folder for date {date} found yet. Assuming fresh start.")
        return []

    # List all topics (folders) inside the date folder
    try:
        results = service.files().list(
            q=f"'{date_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
            fields='files(name)'
        ).execute()
        folders = results.get('files', [])
        
        titles = []
        for f in folders:
            name = f['name']
            # Pattern: News-HHMM-idx-Title
            match = re.search(r'News-\d{4}-\d{2}-(.+)', name)
            if match:
                titles.append(match.group(1).replace('-', ' '))
            else:
                titles.append(name)
        
        if titles:
            print(f"   📈 Found {len(titles)} existing topics today: {', '.join(titles[:3])}...")
        return titles
    except Exception as e:
        print(f"   ⚠️ Error listing topics: {e}")
        return []

def filter_topics_with_ai(new_topics, existing_titles):
    """Uses LLM to aggressively deduplicate and enforce category limits (max 3 per category)."""
    if not new_topics: return []
    
    print(f"🧠 Filtering {len(new_topics)} candidates against {len(existing_titles)} existing items today...")
    
    final_topics = []
    category_counts = {} # e.g., {"Sports": 1, "Technology": 2}
    
    # Build text representation of existing work for prompt context
    processed_context = "\n".join([f"- {t}" for t in existing_titles]) if existing_titles else "No topics processed yet today."

    for topic in new_topics:
        title = topic['title']
        desc = topic['description']
        
        prompt = f"""You are a content curator. I have a list of topics already processed today and a NEW candidate news topic.
        
        EXISTING TOPICS PROCESSED TODAY:
        {processed_context}
        
        NEW CANDIDATE TOPIC:
        Title: {title}
        Description: {desc}
        
        YOUR TASK:
        1. Categorize this NEW topic (e.g., Sports, Technology, Politics, Entertainment, Science, etc.).
        2. Determine if this NEW topic is "similar or redundant" compared to the EXISTING topics. 
           Be AGGRESSIVE: If it covers the same event, person, or immediate sub-topic, mark it as similar.
        
        Return your response strictly as JSON:
        {{
            "category": "Category Name",
            "is_similar": true,
            "reason": "Brief explanation"
        }}"""
        
        try:
            response = generate_text(prompt).strip()
            if '```' in response:
                response = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL).group(1)
            
            res_json = json.loads(response)
            category = res_json.get('category', 'General')
            is_similar = res_json.get('is_similar', False)
            
            if is_similar:
                print(f"   ⏩ Skipping '{title}': Similar to existing ({res_json.get('reason')})")
                continue
            
            # Enforce category limits (max 3)
            current_count = category_counts.get(category, 0)
            if current_count >= 3:
                print(f"   ⏩ Skipping '{title}': Category '{category}' limit reached (3)")
                continue
            
            # Topic approved!
            category_counts[category] = current_count + 1
            final_topics.append(topic)
            print(f"   ✅ Approved '{title}' -> Category: {category} (Current {category}: {category_counts[category]})")
            
            # Add to context for subsequent checks in this same run
            processed_context += f"\n- {title}"
            
        except Exception as e:
            print(f"   ⚠️ AI check failed for '{title}': {e}. Including as fallback.")
            final_topics.append(topic)
            
    return final_topics

def sync_opinions_from_drive(service):
    if not service:
        return ""

    default_opinions = "Write your personal opinions here. They will be included in every video script."
    folder_id = '1tnTb4BjVjOARRKaQjmrse4kddddj9ogj'
    print(f"☁️ Syncing opinions from Google Drive folder: {folder_id}...")

    try:
        service.files().get(fileId=folder_id).execute()

        results = service.files().list(q=f"'{folder_id}' in parents and name='opinions.txt' and trashed=false", spaces='drive', fields='nextPageToken, files(id, name)').execute()
        items = results.get('files', [])

        if not items:
            print("   -> 'opinions.txt' not found inside 'news' folder. Creating default template...")
            from googleapiclient.http import MediaIoBaseUpload
            file_metadata = {'name': 'opinions.txt', 'parents': [folder_id]}
            media = MediaIoBaseUpload(io.BytesIO(default_opinions.encode('utf-8')), mimetype='text/plain')
            service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            return ""
        else:
            file_id = items[0].get('id')
            print("   -> 'opinions.txt' found. Downloading...")
            from googleapiclient.http import MediaIoBaseDownload
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            content_decoded = fh.getvalue().decode('utf-8')
            if content_decoded.strip() and content_decoded.strip() != default_opinions:
                print("   ✅ Custom opinions loaded.")
                return content_decoded
            else:
                print("   ⚠️ Opinions file is empty or default.")
                return ""
    except Exception as e:
        print(f"Failed to sync opinions from drive: {e}")
        return ""



def fetch_top_news(limit=10):
    timestamp = datetime.now().strftime('%H:%M:%S')
    regions = ['US', 'CA']
    all_headlines = []

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    for geo in regions:
        print(f"[{timestamp}] 🌏 Fetching trending news for region: {geo}")

        trends_url = f"https://trends.google.com/trending/rss?geo={geo}"
        try:
            req = urllib.request.Request(trends_url, headers=headers)
            response = urllib.request.urlopen(req, timeout=10)
            xml_data = response.read().decode('utf-8')
            items = re.findall(r'<item>(.*?)</item>', xml_data, re.DOTALL)
            for item in items:
                title = re.search(r'<title>(.*?)</title>', item).group(1).replace('&amp;', '&')
                snippets = re.findall(r'<ht:news_item_snippet>(.*?)</ht:news_item_snippet>', item, re.DOTALL)
                clean_snippets = [re.sub(r'<[^>]+>', '', s).replace('&quot;', '"').replace('&#39;', "'") for s in snippets]

                pic_match = re.search(r'<ht:picture>(.*?)</ht:picture>', item)
                if not pic_match or not pic_match.group(1):
                    pic_match = re.search(r'<ht:news_item_picture>(.*?)</ht:news_item_picture>', item)

                pic_url = pic_match.group(1) if pic_match else ""

                if pic_url:
                    all_headlines.append(f"({geo}) {title}: " + " / ".join(clean_snippets) + f" [Picture: {pic_url}]")
                else:
                    all_headlines.append(f"({geo}) {title}: " + " / ".join(clean_snippets))
            print(f"[{timestamp}] ✅ Got {len(items)} items from {geo} Trends RSS.")
            continue
        except Exception as e:
            print(f"[{timestamp}] ⚠️ {geo} Trends RSS failed ({e}). Trying fallback...")

        news_url = f"https://news.google.com/rss/search?q=trending+news+{geo}&hl=en-US&gl={geo}&ceid={geo}:en"
        try:
            req = urllib.request.Request(news_url, headers=headers)
            response = urllib.request.urlopen(req, timeout=10)
            xml_data = response.read().decode('utf-8')
            items = re.findall(r'<item>(.*?)</item>', xml_data, re.DOTALL)
            for item in items[:15]:
                title = re.search(r'<title>(.*?)</title>', item).group(1).replace('&amp;', '&')
                pic_url = ""
                desc_match = re.search(r'<description>(.*?)</description>', item)
                if desc_match:
                    img_match = re.search(r'img[^>]+src=["\'](.*?)["\']', desc_match.group(1))
                    if img_match:
                        pic_url = img_match.group(1)

                if pic_url:
                    all_headlines.append(f"({geo}) {title} [Picture: {pic_url}]")
                else:
                    all_headlines.append(f"({geo}) {title}")
            print(f"[{timestamp}] ✅ Got {len(items[:15])} fallback items for {geo}.")
        except Exception as e:
            print(f"[{timestamp}] ❌ Failed to fetch news for {geo}: {e}")

    if not all_headlines:
        print(f"[{timestamp}] ❌ No headlines found across any regions.")
        return []

    print(f"[{timestamp}] 🧠 Grouping {len(all_headlines)} headlines across regions into {limit} top topics...")
    try:
        group_prompt = f"Given these raw news headlines from different regions (US, CA), group similar stories together and identify the TOP {limit} most significant and distinct trending topics globally. For each topic, provide a Title, a 2-sentence summary description, and if any of the source headlines included a [Picture: URL], include that URL in a 'picture' field. Format your response strictly as a JSON list of objects: [{{'title': '...', 'description': '...', 'picture': '...'}}, ...]\n\nHeadlines:\n" + "\n".join(all_headlines[:50])

        grouping_json = generate_text(group_prompt).strip()
        if '```' in grouping_json:
            grouping_json = re.search(r'```(?:json)?\s*(.*?)\s*```', grouping_json, re.DOTALL).group(1)

        grouping_json = re.sub(r',\s*([\]}])', r'\1', grouping_json)

        final_topics = json.loads(grouping_json)
        for topic in final_topics:
            topic['link'] = "https://news.google.com"
            topic['pubDate'] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")

        print(f"[{timestamp}] ✅ Unified grouping successful. Found {len(final_topics)} global topics.")
        return final_topics
    except Exception as e:
        print(f"[{timestamp}] ❌ Grouping failed: {e}")
        return [{"title": h.split(': ')[0], "description": h, "link": "", "pubDate": ""} for h in all_headlines[:limit]]

def generate_text(prompt):
    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": "gemini-3-flash",
        "messages": [
            {"role": "system", "content": "You are an expert video script writer producing engaging scripts for YouTube Shorts/TikTok."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        print(f"Calling {url} for text generation...")
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
        response = urllib.request.urlopen(req)
        result = json.loads(response.read().decode('utf-8'))
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"Text generation failed: {e}")
        sys.exit(1)

def generate_image(prompt, output_file, reference_image_url=None):
    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    messages = []
    if reference_image_url:
        try:
            img_req = urllib.request.Request(reference_image_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(img_req) as response:
                image_data = response.read()
                base64_img = base64.b64encode(image_data).decode('utf-8')
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Generate a cinematic 16:9 widescreen aspect ratio image. Re-create the provided image in this style and context: {prompt}. Return the image data as a base64 string."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_img}"
                                }
                            }
                        ]
                    }
                ]
        except Exception as e:
            print(f"Failed to fetch reference image ({e}), falling back to text prompt...")
            messages = [{"role": "user", "content": f"Generate a cinematic 16:9 widescreen aspect ratio image for: {prompt}. Return the image data as a base64 string."}]
    else:
        messages = [{"role": "user", "content": f"Generate a cinematic 16:9 widescreen aspect ratio image for: {prompt}. Return the image data as a base64 string."}]

    payload = {
        "model": "gemini-3.1-flash-image-16-9",
        "messages": messages
    }

    try:
        print(f"Calling {url} with model gemini-3.1-flash-image for image generation...")
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
        response = urllib.request.urlopen(req)
        result = json.loads(response.read().decode('utf-8'))
        content = result['choices'][0]['message']['content']

        base64_matches = re.findall(r'[A-Za-z0-9+/]{100,}', content)

        if not base64_matches:
            base64_matches = re.findall(r'base64,([A-Za-z0-9+/=]+)', content)

        if base64_matches:
            base64_str = max(base64_matches, key=len)
            print(f"Detected base64 image data ({len(base64_str)} chars). Decoding...")

            missing_padding = len(base64_str) % 4
            if missing_padding:
                base64_str += '=' * (4 - missing_padding)

            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))

            if not output_file.lower().endswith('.png'):
                output_file = os.path.splitext(output_file)[0] + '.png'

            image.save(output_file, "PNG")
            print(f"✅ Image saved successfully to {output_file}")
            return True
        else:
            url_match = re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
            if url_match:
                image_url = url_match.group(0)
                print(f"Downloading image from {image_url}...")
                img_req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(img_req) as response:
                    image_data = response.read()
                    image = Image.open(io.BytesIO(image_data))
                    if not output_file.lower().endswith('.png'):
                        output_file = os.path.splitext(output_file)[0] + '.png'
                    image.save(output_file, "PNG")
                return True

            print(f"No image data or URL found in response.")
            raise Exception("No image data found")
    except Exception as e:
        print(f"Image generation failed: {e}")
        raise e

def generate_image_with_retry(prompt, output_file, reference_image_url=None, retries=MAX_RETRIES):
    for attempt in range(1, retries + 1):
        try:
            return generate_image(prompt, output_file, reference_image_url)
        except Exception as e:
            print(f"[Attempt {attempt}/{retries}] Image generation failed: {e}")
            if attempt < retries:
                time.sleep(2)  # optional: wait before retrying
            else:
                print(f"⚠️ Skipping image after {retries} failed attempts.")
                return None  # skip and continue pipeline

def generate_audio(text, output_file, voice="alloy"):
    url = f"{API_BASE_URL}/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": "tts-1",
        "input": text,
        "voice": voice
    }

    try:
        print(f"Calling {url} for audio generation...")
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
        response = urllib.request.urlopen(req)
        with open(output_file, 'wb') as f:
            f.write(response.read())
        print(f"✅ Audio saved successfully to {output_file}")
        return True
    except Exception as e:
        print(f"Audio generation via API failed: {e}. Attempting edge-tts fallback...")

        voice_map = {
            "alloy": "en-US-AriaNeural",
            "echo": "en-US-GuyNeural",
            "fable": "en-GB-SoniaNeural",
            "onyx": "en-US-ChristopherNeural",
            "nova": "en-US-NatashaNeural",
            "shimmer": "en-US-JennyNeural"
        }
        edge_voice = voice_map.get(voice, "en-US-AriaNeural")

        try:
            import edge_tts
        except ImportError:
            os.system("pip install edge-tts")
            import edge_tts

        import asyncio
        async def _generate_edge_audio():
            clean_text = text.strip()
            clean_text = re.sub(r'[*_`#]', '', clean_text)
            if not clean_text: return False

            for attempt in range(3):
                try:
                    communicate = edge_tts.Communicate(clean_text, edge_voice)
                    await communicate.save(output_file)
                    if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
                        return True
                    print(f"   ⚠️ Edge-TTS attempt {attempt+1} produced empty file. Retrying...")
                except Exception as e:
                    print(f"   ⚠️ Edge-TTS attempt {attempt+1} failed: {e}. Retrying...")
                await asyncio.sleep(2)
            return False

        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            success = False
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    success = executor.submit(lambda: asyncio.run(_generate_edge_audio())).result()
            else:
                success = loop.run_until_complete(_generate_edge_audio())

            if success:
                print(f"✅ Audio saved successfully via edge-tts to {output_file}")
                return True
            else:
                raise Exception("Exhausted retries or empty audio produced")
        except Exception as edge_e:
            print(f"Edge-tts generation failed: {edge_e}")
            return False

def download_bg_music(style, output_file):
    try:
        import yt_dlp
    except ImportError:
        os.system("pip install yt-dlp")
        import yt_dlp

    print(f"🎵 Downloading background music for style: {style}")

    cookie_file = "cookies.txt"
    browser_session = os.path.expanduser("~/.browser-session")

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_file.rsplit('.', 1)[0],
        'quiet': True,
        'noplaylist': True,
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'logtostderr': False,
    }

    if os.path.exists(cookie_file):
        print(f"   -> Using standalone {cookie_file} for YouTube authentication...")
        ydl_opts['cookiefile'] = cookie_file
    elif os.path.exists(browser_session):
        print(f"   -> Injecting cookies from {browser_session} to bypass bot detection...")
        ydl_opts['cookiesfrombrowser'] = ('chrome', browser_session)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_query = f"ytsearch1:{style} instrumental background music no copyright creative commons royalty free"
            ydl.extract_info(search_query, download=True)

        print(f"✅ Background music downloaded to {output_file}")
        return True
    except Exception as e:
        print(f"Failed to download background music: {e}")
        return False

def combine_audio(voice_file, bg_file, output_file):
    print(f"🎛️ Combining voice and background music...")
    cmd = [
        "ffmpeg", "-y",
        "-i", voice_file,
        "-i", bg_file,
        "-filter_complex", "[1:a]volume=0.08[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2",
        output_file
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Final audio saved to {output_file}")
        return True
    except Exception as e:
        print(f"Audio combine failed: {e}")
        return False

def clean_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def pick_visual_style(title):
    """
    Decides whether to use artistic or photographic mode based on ART_PERCENTAGE,
    then returns (is_artistic, image_style, mode_desc).

    - Artistic mode: cycles through POPULAR_ARTISTIC_STYLES IN ORDER (round-robin).
    - Photographic mode: picks randomly from photographic categories (unchanged behaviour).
    """
    global _artistic_style_index

    is_artistic_mode = random.random() < ART_PERCENTAGE

    if is_artistic_mode:
        # Round-robin through the ordered popular artistic styles list
        style = POPULAR_ARTISTIC_STYLES[_artistic_style_index % len(POPULAR_ARTISTIC_STYLES)]
        _artistic_style_index += 1
        mode_desc = "iconic ARTISTIC visual style"
        print(f"   🎨 Artistic mode → style #{_artistic_style_index}: {style}")
        return True, style, mode_desc
    else:
        # Photographic mode stays fully random (original behaviour)
        categories = "National Geographic, Cinematic 4k Film Still, Magnum Photography (raw/journalistic), Drone Perspective, Kodak Portra 400, or Macro Tech Photography"
        mode_desc = "professional PHOTOGRAPHIC visual style"
        style_query = f"Given the news story: '{title}', pick the most fitting {mode_desc} from these categories: {categories}. Return ONLY the resulting style phrase (under 10 words) and NOTHING else."
        style = generate_text(style_query).strip().strip('"\'-\n')
        if not style:
            style = "Cinematic National Geographic style"
        print(f"   📷 Photographic mode → style: {style}")
        return False, style, mode_desc

def main():
    service = get_drive_service()
    user_opinions = ""
    existing_titles = []
    
    if service:
        user_opinions = sync_opinions_from_drive(service)
        existing_titles = get_todays_processed_titles(service)
    else:
        print("⚠️ No Drive service available. Skipping history check and opinions sync.")

    now = get_pacific_time()
    year_str = now.strftime("%Y")
    month_str = now.strftime("%m")
    date_str = now.strftime("%Y-%m-%d")

    base_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'news'))
    os.makedirs(base_dir, exist_ok=True)

    # Use TOPIC_LIMIT from env (default 10) to have enough candidates after filtering
    raw_news = fetch_top_news(limit=TOPIC_LIMIT)
    news_items = filter_topics_with_ai(raw_news, existing_titles)

    if not news_items:
        print("📭 No new topics to process after filtering and limits.")
        return

    for idx, item in enumerate(news_items):
        title = item['title']
        print(f"\n============================")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing [{idx+1}/{len(news_items)}]: {title}")
        print(f"============================")

        clean_title = clean_filename(title)[:50]
        start_hhmm = now.strftime("%H%M")

        year_month_dir = os.path.join(base_dir, year_str, month_str, date_str)
        if not os.path.exists(year_month_dir):
            os.makedirs(year_month_dir)

        folder_name = f"News-{start_hhmm}-{idx+1:02d}-{clean_title}"
        project_dir = os.path.join(year_month_dir, folder_name)

        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

        opinions_block = ""
        if user_opinions:
            opinions_block = f"\nUSER OPINIONS & PERSPECTIVE TO INCLUDE:\n{user_opinions}\n\nCRITICAL: You MUST weave these opinions seamlessly into the dialogue of the script. At least one of the characters (e.g., the Anchor or the Expert) must embody this perspective and express these views while discussing the story.\n"

        script_prompt = f"""Write a viral video script about the following news story.
Include a strong hook, the main story, and a call to action.
To make it highly engaging, you MUST invent realistic comments, reactions, or interview snippets from roles relevant to the article (e.g., industry experts, eyewitnesses, politicians, or online public reactions).
The script MUST contain enough spoken dialogue to last AT LEAST 3 FULL MINUTES (approx 450-600 words).
You MUST use multiple roles/characters (e.g., Studio Anchor, Relevant Expert, Bystander) and assign a different voice to each role from this list: alloy, echo, fable, onyx, nova, shimmer.
{opinions_block}
CRITICAL FORMATTING: You MUST format the output EXACTLY like a 'lyrics_with_prompts.md' file. Interleave timestamps (in 8-second intervals), VIDEO generation prompts describing motion in brackets, and the spoken script text.
It MUST look exactly like this structure:

Song Title: {title}
Style: Interleaved (Batch Processed)

> 00:00-00:08 [Video Prompt] A dynamic, slow-motion tracking shot showing...
[Voice: onyx] Breaking news tonight as the situation unfolds...

> 00:08-00:16 [Video Prompt] Fast zoom onto the reporter standing in front of...
[Voice: nova] That's right, experts are now saying...

...continue this 8-second chunk sequence until the FULL 3 minutes is reached.

Title: {title}
Details: {item['description']}"""

        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✍️ Generating 3-minute video script...")
        script_content = generate_text(script_prompt)

        with open(os.path.join(project_dir, "lyrics_with_prompts.md"), "w", encoding='utf-8') as f:
            if "Song Title:" not in script_content:
                f.write(f"Song Title: {title}\nStyle: Interleaved (Batch Processed)\n\n{script_content}\n")
            else:
                f.write(f"{script_content}\n")

        # ── Visual style selection ──────────────────────────────────────────
        # ART_PERCENTAGE controls artistic vs photographic split.
        # Artistic styles cycle in fixed order via POPULAR_ARTISTIC_STYLES.
        # Photographic styles remain fully random (original behaviour).
        is_artistic_mode, image_style, mode_desc = pick_visual_style(title)

        anthro_chance = random.random()
        anthro_modifier = ""
        if anthro_chance < ANTHROPOMORPHIC_PERCENTAGE:
            anthro_types = ["anthropomorphic animals", "cute anthropomorphic cats and dogs", "anthropomorphic fantasy creatures", "anthropomorphic animals in business attire", "whimsical anthropomorphic wildlife"]
            selected_anthro = random.choice(anthro_types)
            anthro_modifier = f" KEY DIRECTIVE: Re-imagine all human subjects and participants as {selected_anthro}."

        image_prompt = f"A compelling, high-quality cover thumbnail image without any text for a news story titled: '{title}'. Style: {image_style}. Details: {item['description']}.{anthro_modifier}"

        picture_url = item.get('picture', '')
        if picture_url:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🖼️ Generating cover (Ref: {picture_url[:30]}...) with style: {image_style}{' [Anthro]' if anthro_modifier else ''}")
            result = generate_image_with_retry(image_prompt, os.path.join(project_dir, "cover.png"), picture_url)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🖼️ Generating cover (No Ref) with style: {image_style}{' [Anthro]' if anthro_modifier else ''}")
            result = generate_image_with_retry(image_prompt, os.path.join(project_dir, "cover.png"))

        if result is None:
            print(f"⚠️ Image generation failed for story: {title}. Skipping...")
            continue

        char_prompt = f"Write a detailed visual description of the main characters, subjects, and the environmental setting suitable for the cover image of a news story titled: '{title}'. Describe their appearance, clothing, the mood of the location, and lighting. Make it highly descriptive.\n\nDetails: {item['description']}"
        char_content = generate_text(char_prompt).strip()
        with open(os.path.join(project_dir, "charactor.md"), "w", encoding='utf-8') as f:
            f.write(f"# Cover Image & Environment Description\n\n{char_content}\n")

        ref_dir = os.path.join(project_dir, "references")
        if not os.path.exists(ref_dir):
            os.makedirs(ref_dir)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎥 Searching for reference video/photos...")
        try:
            import yt_dlp

            cookie_file = "cookies.txt"
            browser_session = os.path.expanduser("~/.browser-session")

            ref_ydl_opts = {
                'format': 'best',
                'outtmpl': os.path.join(ref_dir, 'ref_video_%(id)s.%(ext)s'),
                'quiet': True,
                'noplaylist': True,
                'max_downloads': 1
            }

            if os.path.exists(cookie_file):
                print(f"   -> Using {cookie_file} for reference video authentication...")
                ref_ydl_opts['cookiefile'] = cookie_file
            elif os.path.exists(browser_session):
                print(f"   -> Using browser session for reference video authentication...")
                ref_ydl_opts['cookiesfrombrowser'] = ('chrome', browser_session)

            with yt_dlp.YoutubeDL(ref_ydl_opts) as ydl:
                ydl.extract_info(f"ytsearch1:{title} news footage", download=True)
        except Exception as e:
            print(f"   ⚠️ Could not automatically download reference video: {e}")

        voice_file = os.path.join(ref_dir, "full_voice.mp3")
        bg_file = os.path.join(ref_dir, "bg_music.mp3")
        final_audio = os.path.join(project_dir, "combined_audio.mp3")

        import re
        blocks = re.findall(r'\[Voice:\s*(\w+)\]\s*(.*?)(?=\n>|\Z)', script_content, re.DOTALL | re.IGNORECASE)

        if not blocks:
            clean_text = re.sub(r'\[.*?\]', '', script_content)
            clean_text = re.sub(r'>.*?$', '', clean_text, flags=re.MULTILINE)
            clean_text = clean_text.replace(f'Song Title: {title}', '').replace('Style: Interleaved (Batch Processed)', '').strip()
            if not clean_text: clean_text = title
            generate_audio(clean_text[:4000], voice_file, voice='alloy')
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎙️ Generating {len(blocks)} multi-voice audio chunks for the full 3-minute script...")
            chunk_files = []
            for i, (voice_name, text) in enumerate(blocks):
                chunk_text = text.strip()
                if not chunk_text: continue

                chunk_text = chunk_text.replace('**', '').replace('*', '')

                voice_name = voice_name.lower()
                if voice_name not in ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']:
                    voice_name = 'alloy'

                chunk_file = os.path.join(ref_dir, f"chunk_{i:03d}.mp3")
                if generate_audio(chunk_text[:4000], chunk_file, voice=voice_name):
                    chunk_files.append(chunk_file)

            if chunk_files:
                concat_list = os.path.join(ref_dir, "concat.txt")
                with open(concat_list, "w", encoding='utf-8') as cf:
                    for cf_file in chunk_files:
                        cf.write(f"file '{os.path.basename(cf_file)}'\n")

                print("🔗 Concatenating audio chunks...")
                cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", voice_file]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    print(f"Concat failed: {e}")
                    import shutil
                    shutil.copy(chunk_files[0], voice_file)

                for cf_file in chunk_files:
                    try: os.remove(cf_file)
                    except: pass
                if os.path.exists(concat_list):
                    os.remove(concat_list)

        bg_style_prompt = f"Give a 2-word genre description for background music suited for a news story about: '{title}' (e.g. 'tense cinematic', 'upbeat tech', 'somber ambient'). Return ONLY the genre description."
        bg_style = generate_text(bg_style_prompt).strip().strip('"\'-\n')

        if download_bg_music(bg_style, bg_file):
            if not combine_audio(voice_file, bg_file, final_audio):
                import shutil
                shutil.copy(voice_file, final_audio)
        else:
            import shutil
            shutil.copy(voice_file, final_audio)

        with open(os.path.join(ref_dir, "references.txt"), "w", encoding='utf-8') as f:
            f.write(f"Title: {title}\nURL: {item.get('link', '')}\nDate: {item.get('pubDate', '')}")

        upload_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'upload_to_drive.py'))

        print(f"Checking for upload script at: {upload_script_path}")
        if os.path.exists(upload_script_path):
            print(f"🚀 Starting the upload phase to Google Drive for {folder_name}...")
            cmd = [sys.executable, upload_script_path, project_dir]
            subprocess.run(cmd)
            print(f"✅ Upload phase finished for {folder_name}.")
        else:
            print(f"⚠️ Upload script NOT FOUND. Skipping Drive upload.")

        time.sleep(2)

    print(f"\n✅ All local processing complete! Projects saved in: {base_dir}")
    print(f"============================================================")

if __name__ == "__main__":
    main()