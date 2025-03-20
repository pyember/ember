settings = dict(
    sqlite=dict(provider='sqlite', filename=':memory:'),
    mysql=dict(provider='mysql', host="localhost", user="ponytest", passwd="ponytest", db="ponytest"),
    postgres=dict(provider='postgres', host='localhost', user='ponytest', password='ponytest', database='ponytest'),
    cockroach=dict(provider='cockroach', user='root', host='localhost', port=26257, database='ponytest', sslmode='disable'),
    oracle=dict(provider='oracle', user='c##ponytest', password='ponytest', dsn='localhost/orcl')
)['mysql']
