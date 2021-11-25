# Some code adapted from https://github.com/IRLL/HIPPO_Gym/
import asyncio, websockets, json, sys
import time
from multiprocessing import Process, Pipe

from coopihc.interactiontask import PipeTaskWrapper
from coopihc.bundle.wrappers import PipedTaskBundleWrapper


# like functools.partial, but with arguments added from the back
def partialback(func, *extra_args):
    def wrapper(*args):
        args = list(args)
        args.extend(extra_args)
        return func(*args)

    return wrapper


class Server:
    def __init__(self, bundle, taskwrapper, address="localhost", port=4000):
        self.start_server = websockets.serve(
            partialback(self.bundlehandler, bundle, taskwrapper), address, port
        )
        self.user = None
        self.bundle = bundle

    def start(self):
        asyncio.get_event_loop().run_until_complete(self.start_server)
        asyncio.get_event_loop().run_forever()

    async def bundlehandler(self, websocket, path, bundle, taskwrapper):
        """
        On websocket connection, starts a new userTrial in a new Process.
        Then starts async listeners for sending and recieving messages.
        """
        await self.register(websocket)
        bundlepipeup, bundlepipedown = Pipe()
        process = Process(
            target=PipedTaskBundleWrapper, args=(bundle, taskwrapper, bundlepipedown)
        )
        process.start()
        consumerTask = asyncio.ensure_future(
            self.consumer_handler(websocket, bundlepipeup)
        )
        producerTask = asyncio.ensure_future(
            self.producer_handler(websocket, bundlepipeup)
        )
        done, pending = await asyncio.wait(
            [consumerTask, producerTask], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await websocket.close()
        return

    async def register(self, websocket):
        self.user = websocket
        print("new task connected: {}".format(str(websocket)))

    async def consumer_handler(self, websocket, pipe):
        async for message in websocket:
            print("received message {}".format(message))
            # print("json", [[key, value, type(value)] for key, value in json.loads(message).items()])
            pipe.send(json.loads(message))

    async def producer_handler(self, websocket, pipe):
        """
        Look for messages to send from the Bundle.
        asyncio.sleep() is required to make this non-blocking
        default sleep time is (0.01) which creates a maximum framerate of
        just under 100 frames/s. For faster framerates decrease sleep time
        however be aware that this will affect the ability of the
        consumer_handler function to keep up with messages from the websocket
        and may cause poor performance if the web-client is sending a high volume
        of messages.
        """
        done = False
        while True:
            done = await self.producer(websocket, pipe)
            await asyncio.sleep(0.01)
        return

    async def producer(self, websocket, pipe):
        """
        Check userTrial process pipe for messages to send to websocket.
        If userTrial is done, send final message to websocket and return
        True to tell calling functions that userTrial is complete.
        """
        if pipe.poll():
            message = pipe.recv()
            if message == "done":
                await websocket.send(json.dumps({"type": "done"}))
                return True
            else:
                print("sending message: \t {}".format(json.dumps(message)))
                await websocket.send(json.dumps(message))
        return False
