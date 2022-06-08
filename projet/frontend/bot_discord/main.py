import os
import requests
import discord

from discord.ext import commands
from dotenv import load_dotenv

# Get bot token
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

# Init Bot
bot = commands.Bot(command_prefix='!')
bot.enable_automatic_description = True
bot.language = "EN"
bot.supported_languages = ["BG", "CS", "DA", "DE", "EL", "ES", "FI", "FR", "HU",
                               "ID", "IT", "JA", "LT", "LV", "NL", "PL", "PT-PT", "PT-BR", "RO", "RU", "SK", "SL", "SV", "TR", "ZH"]

# Command test
@bot.command(name='hi', help='Say hi', scope=847072561816141884,)
async def hi(ctx):
    ctx.send("Hi")

# Event called when bot is ready 
@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')

@bot.event
async def on_message(message):
    # Describe automatically images
    if bot.enable_automatic_description and len(message.attachments) == 1 and message.attachments[0].content_type.startswith('image/'):
        await request_from_url(message, message.attachments[0].url)

    ### COMMANDS ###
    # !describeUrl : Describe an image from an url
    elif message.content.startswith('!describeUrl'):
        await request_from_url(message, message.content.split(" ")[1])

    # !disable : Disable automatic image description
    elif message.content.startswith('!disable'):
        bot.enable_automatic_description = False
        
        embedMsg = discord.Embed(title="", description="Automatic description disabled", color=0xff0000)
        await message.reply(embed = embedMsg)

    # !enable : Enable automatic image description
    elif message.content.startswith('!enable'):
        bot.enable_automatic_description = True

        embedMsg = discord.Embed(title="", description="Automatic description enabled", color=0xff0000)
        await message.reply(embed = embedMsg)

    # !about : Everything you have to know about me
    elif message.content.startswith('!about'):
        embedMsg = discord.Embed(title="About me :", description="", color=0xff0000)
        embedMsg.add_field(name="Website", value="https://the-describers.netlify.app/", inline=False)
        embedMsg.add_field(name="Github", value="https://github.com/iFairPlay22/DescribeImage", inline=False)
        embedMsg.add_field(name="Supported languages", value="BG, CS, DA, DE, EL, ES, FI, FR, HU, ID, IT, JA, LT, LV, \nNL, PL, PT-PT, PT-BR, RO, RU, SK, SL, SV, TR, ZH", inline=False)
        embedMsg.add_field(name="Authors", value="• Fabien Courtois\n• Loïc Fournier\n• Lucas Billard\n• Ewen Bouquet", inline=False)
        await message.reply(embed = embedMsg)

    # !commands : List every commands available
    elif message.content.startswith('!commands'):
        embedMsg = discord.Embed(title="Commands list :", description="", color=0xff0000)
        embedMsg.add_field(name="describeUrl", value="Describe an image from an url", inline=False)
        embedMsg.add_field(name="disable", value="Disable automatic image description", inline=False)
        embedMsg.add_field(name="enable", value="Enable automatic image description", inline=False)
        embedMsg.add_field(name="about", value="Everything you have to know about me", inline=False)
        embedMsg.add_field(name="language", value="Switch to specified language", inline=False)
        embedMsg.add_field(name="commands", value="List every commands available", inline=False)
        await message.reply(embed = embedMsg)
    # !language : Switch to specified language 
    elif message.content.startswith('!language'):
        tab = message.content.split(" ")
        if(len(tab) == 2 and tab[1] in bot.supported_languages):
            bot.language = tab[1]
            embedMsg = discord.Embed(title="", description="Language changed to " + bot.language, color=0xff0000)
            await message.reply(embed = embedMsg)
        else:
            embedMsg = discord.Embed(title="", description="", color=0xff0000)
            embedMsg.add_field(name="Supported languages", value="BG, CS, DA, DE, EL, ES, FI, FR, HU, ID, IT, JA, LT, LV, \nNL, PL, PT-PT, PT-BR, RO, RU, SK, SL, SV, TR, ZH", inline=False)
            await message.reply(embed = embedMsg)
    else:
        return

# Call backend API to get image description from file url
async def request_from_url(message, url):
    await message.channel.send('Processing ...')

    res = requests.post('https://www.loicfournier.fr/iadecode/from_url/'+bot.language, json={'file': url})
        
    if(res.status_code == 200):
        embedMsg = discord.Embed(title="", description=res.json()['message'].capitalize(), color=0xff0000)
        await message.reply(embed = embedMsg)
    else:
        embedMsg = discord.Embed(title="", description="Error : verify your url", color=0xff0000)
        await message.reply(embed = embedMsg)
    return

bot.run(TOKEN)