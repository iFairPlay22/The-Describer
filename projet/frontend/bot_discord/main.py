import os
import requests

from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

bot = commands.Bot(command_prefix='/')

@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')

@bot.event
async def on_message(message):
    #commands
    if message.content.startswith('!describeUrl'):
        await message.channel.send('Processing ...')
        
        url = message.content.split(" ")[1]
        res = requests.post('http://217.160.10.8:80/iadecode/from_url', json={'file': url})
        
        if(res.status_code == 200):
            await message.channel.send(res.json()['message'])
        else:
            await message.channel.send('Error : verify your url')

bot.run(TOKEN)