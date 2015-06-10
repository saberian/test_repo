import MySQLdb as mdb

con = mdb.connect('localhost', 'testuser', 'test623', 'testdb');

'''with con:
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS Writers")
    cur.execute("CREATE TABLE Writers(Id INT PRIMARY KEY AUTO_INCREMENT, \
                 Name VARCHAR(25))")
    cur.execute("INSERT INTO Writers(Name) VALUES('Jack London')")
    cur.execute("INSERT INTO Writers(Name) VALUES('Honore de Balzac')")
    cur.execute("INSERT INTO Writers(Name) VALUES('Lion Feuchtwanger')")
    cur.execute("INSERT INTO Writers(Name) VALUES('Emile Zola')")
    cur.execute("INSERT INTO Writers(Name) VALUES('Truman Capote')")'''


with con: 

    #cur = con.cursor()
    cur = con.cursor(mdb.cursors.DictCursor)
    cur.execute("SELECT * FROM Writers")
    desc = cur.description
    print "%s %3s" % (desc[0][0], desc[1][0])
    rows = cur.fetchall()
    for row in rows:
        #print row
        print [row["Id"], row["Name"]]
    '''for i in range(cur.rowcount):    
        row = cur.fetchone()
        print row[0], row[1]'''