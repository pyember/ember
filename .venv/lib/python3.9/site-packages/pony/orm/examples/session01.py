from pony.orm import *

db = Database('sqlite', ':memory:')


class Person(db.Entity):
    name = Required(str)
    age = Required(int)


db.generate_mapping(create_tables=True)

with db_session:
    p1 = Person(name='John', age=20)
    p2 = Person(name='Mike', age=18)


x = 10
y = 15

with db_session:
    q = select(p.name for p in Person if p.name.startswith('J') and p.age < x + y)
    q.show()





























