import discord
from credentials import DISCORD_TOKEN

client = discord.Client()

async def generate_route(message: str):
    prompt = message.content[10:]


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

@client.event
async def on_message(message):

    if message.content.startswith("!generate"):
        await generate_route(message)

client.run(DISCORD_TOKEN)