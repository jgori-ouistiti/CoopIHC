from core.bundle import Bundle
import asyncio, websockets

class Server():
    clients = set()

    async def register(self, websocket):
        self.clients.add(websocket)
        print('{} connects'.format(websocket))

    async def unregister(self, websocket):
        self.clients.remove(websocket)
        print("{} disconnects".format(websocket))

    async def send_to_clients(self, message):
        if self.clients:
            await asyncio.wait([client.send(message) for client in self.clients])

    async def websocket_handler(self, websocket, uri):
        await self.register(websocket)
        try:
            await self.distribute(websocket)
        finally:
            await self.unregister(websocket)

    async def distribute(self, websocket):
        async for message in websocket:
            await self.send_to_clients(message)

class WebSocketBundle(Bundle):
    def __init__(self, bundle):
        self.bundle = bundle
        server = Server()
        start_server = websockets.serve(server.websocket_handler, 'localhost', 4000)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(start_server)

    def reset(self, dic = {}):
        return self.bundle.reset(dic = dic)

    def step(self, *args, **kwargs):
        return self.bundle.step(*args, **kwargs)

    def render(self, mode, *args, **kwargs):
        return self.bundle.render(mode, *args, **kwargs)

    def close(self):
        return self.bundle.close()
