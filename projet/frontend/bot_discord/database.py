import sqlite3

#################
##### USERS #####
#################

# Check if user exists in database
def userExists(user):
    # Connect to database
    conn = sqlite3.connect('database.db')
    conn.cursor()

    # Set language for the user table in the database
    sql = "SELECT Id FROM Users WHERE Id='" + str(user.id) + "';"
    row = conn.execute(sql).fetchone()

    #Close connection
    conn.close()

    # Return language
    if row:
        return True
    else:
        return False

# Update language for a specific user
def updateUserLanguage(user, language):
    # Connect to database
    conn = sqlite3.connect('database.db')
    conn.cursor()

    # Set language for the user table in the database
    sql = "UPDATE Users SET Language='" + language + "' WHERE Id='" + str(user.id) + "';"
    conn.execute(sql)
    conn.commit()

    #Close connection
    conn.close()

# Set language for a specific user
def setUserLanguage(user, language):
    # Connect to database
    conn = sqlite3.connect('database.db')
    conn.cursor()
    
    # Set language for the user table in the database
    sql = "INSERT INTO Users ('Id','Pseudo','Language') VALUES('" + str(user.id) + "','" + user.name + "','" + language + "');"
    conn.execute(sql)
    conn.commit()
    
    #Close connection
    conn.close()

# Get language for a specific user
def getUserLanguage(user):

    # Connect to database
    conn = sqlite3.connect('database.db')
    conn.cursor()

    # Set language for the user table in the database
    sql = "SELECT Language FROM Users WHERE Id='" + str(user.id) + "';"
    row = conn.execute(sql).fetchone()

    #Close connection
    conn.close()

    # Return language
    if row:
        return row[0]
    else:
        return "EN"

####################
##### CHANNELS #####
####################

# Check if channel exists in database
def channelExists(channel):
    # Connect to database
    conn = sqlite3.connect('database.db')
    conn.cursor()

    # Set language for the user table in the database
    sql = "SELECT Id FROM Channels WHERE Id='" + str(channel.id) + "';"
    row = conn.execute(sql).fetchone()

    #Close connection
    conn.close()

    # Return language
    if row:
        return True
    else:
        return False

# Update bot enable for a specific channel
def updateChannelBotEnable(channel, botEnable):
    # Connect to database
    conn = sqlite3.connect('database.db')
    conn.cursor()

    # Set language for the user table in the database
    sql = "UPDATE Channels SET BotEnable=" + botEnable + " WHERE Id='" + str(channel.id) + "';"
    conn.execute(sql)
    conn.commit()

    #Close connection
    conn.close()

# Set bot enable for a specific channel
def setChannelBotEnable(channel, botEnable):
    # Connect to database
    conn = sqlite3.connect('database.db')
    conn.cursor()
    
    # Set language for the user table in the database
    sql = "INSERT INTO Channels ('Id','Name','BotEnable','Guilds_Id') VALUES('" + str(channel.id) + "','" + channel.name + "'," + botEnable + ",'" + str(channel.guild.id) + "');"
    conn.execute(sql)
    conn.commit()
    
    #Close connection
    conn.close()

# Get bot enable for a specific channel
def getChannelBotEnable(channel):
    
        # Connect to database
        conn = sqlite3.connect('database.db')
        conn.cursor()
    
        # Get bot enable for the channel table in the database
        sql = "SELECT BotEnable FROM Channels WHERE Id='" + str(channel.id) + "';"
        row = conn.execute(sql).fetchone()
    
        #Close connection
        conn.close()
    
        # Return bot enable
        if row:
            if row[0] == 0:
                return False
            else:
                return True
        else:
            return True

##################
##### GUILDS #####
##################

# Check if guild exists in database
def guildExists(guild):
    # Connect to database
    conn = sqlite3.connect('database.db')
    conn.cursor()

    # Set language for the user table in the database
    sql = "SELECT Id FROM Guilds WHERE Id='" + str(guild.id) + "';"
    row = conn.execute(sql).fetchone()

    #Close connection
    conn.close()

    # Return language
    if row:
        return True
    else:
        return False

# Update bot enable for a specific guild
def updateGuildBotEnable(guild, botEnable):
    # Connect to database
    conn = sqlite3.connect('database.db')
    conn.cursor()

    # Set language for the user table in the database
    sql = "UPDATE Guilds SET BotEnable=" + botEnable + " WHERE Id='" + str(guild.id) + "';"
    conn.execute(sql)
    conn.commit()

    #Close connection
    conn.close()

# Set bot enable for a specific guild
def setGuildBotEnable(guild, botEnable):
    # Connect to database
    conn = sqlite3.connect('database.db')
    conn.cursor()
    
    # Set language for the user table in the database
    sql = "INSERT INTO Guilds ('Id','Name','BotEnable') VALUES('" + str(guild.id) + "','" + guild.name + "'," + botEnable + ");"
    conn.execute(sql)
    conn.commit()
    
    #Close connection
    conn.close()

# Get bot enable for a specific guild
def getGuildBotEnable(guild):
        
            # Connect to database
            conn = sqlite3.connect('database.db')
            conn.cursor()
        
            # Get bot enable for the guild table in the database
            sql = "SELECT BotEnable FROM Guilds WHERE Id='" + str(guild.id) + "';"
            row = conn.execute(sql).fetchone()
        
            #Close connection
            conn.close()
        
            # Return bot enable
            if row:
                if row[0] == 0:
                    return False
                else:
                    return True
            else:
                return True
