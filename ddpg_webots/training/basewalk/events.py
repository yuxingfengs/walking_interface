"""Local event notifications. No agent polling, network calls or training control."""
import base64
from html import escape
import json
import os
from pathlib import Path
import subprocess
import time


def windows_toast(title, message):
    if os.name != 'nt':
        return dict(status='unsupported_platform')
    xml = '<toast><visual><binding template="ToastGeneric"><text>'+escape(title)+'</text><text>'+escape(message)+'</text></binding></visual></toast>'
    encoded_xml = base64.b64encode(xml.encode('utf-8')).decode('ascii')
    script = "$ErrorActionPreference='Stop'; [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null; [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime] | Out-Null; $doc=New-Object Windows.Data.Xml.Dom.XmlDocument; $doc.LoadXml([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('"+encoded_xml+"'))); $toast=[Windows.UI.Notifications.ToastNotification]::new($doc); $notifier=[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Microsoft.Windows.PowerShell'); $notifier.Show($toast)"
    command = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    result = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',command],
        capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
    return dict(status='submitted_to_windows' if result.returncode == 0 else 'failed',
                returncode=result.returncode, error=result.stderr.decode(errors='replace')[-1500:] if result.returncode else '')


def emit(folder, config, kind, title, message, **fields):
    """Persist every event; delivery failure never aborts training."""
    record = dict(kind=kind, title=title, message=message, wall_time=time.time(), **fields)
    folder = Path(folder)
    try:
        with (folder/'alerts.jsonl').open('a',encoding='utf-8') as out:
            out.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    except OSError as exc:
        print(f'[alert log failed] {exc}',flush=True)
    print('[alert] '+json.dumps(dict(title=title,message=message),ensure_ascii=True),flush=True)
    if config.get('notifications',{}).get('desktop',False):
        try:
            delivery = windows_toast(title,message)
        except Exception as exc:
            delivery = dict(status='failed',error=str(exc))
        try:
            with (folder/'notification_delivery.jsonl').open('a',encoding='utf-8') as out:
                out.write(json.dumps(dict(kind=kind,wall_time=time.time(),**delivery),ensure_ascii=False)+'\n')
        except OSError as exc:
            print(f'[notification status failed] {exc}',flush=True)
