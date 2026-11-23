"""Detached Windows exit handler; waits on a process handle, not training logs."""
import ctypes
from ctypes import wintypes
import json
import sys
from pathlib import Path


def wait_for_exit(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:  # Process already exited.
            return
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if kernel.WaitForSingleObject(handle, 0xFFFFFFFF) != 0:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


if __name__ == '__main__':
    folder = Path(sys.argv[2])
    try:
        wait_for_exit(int(sys.argv[1]))
        if not (folder/'reward_plots/normal_exit.done').exists():
            from reward_plots import plot_rewards
            output = plot_rewards(folder)
            (folder/'exit_plot_status.json').write_text(json.dumps(dict(
                status='completed' if output else 'no_saved_training_data',
                output=str(output) if output else None)), encoding='utf-8')
    except Exception as error:
        (folder/'exit_plot_status.json').write_text(json.dumps(dict(
            status='failed', error=str(error))), encoding='utf-8')
