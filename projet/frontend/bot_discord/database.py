import sqlite3

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
    sql = "INSERT INTO Users VALUES('" + str(user.id) + "','" + user.name + "','" + language + "');"
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