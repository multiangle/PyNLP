import threading
import asyncio

class thd(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        pass

async def cor():
    async with asyncio.sleep(2):
        
