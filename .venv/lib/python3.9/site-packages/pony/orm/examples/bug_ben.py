""""
Traceback (most recent call last):
 File "/crm/mobile_sync_system/handlers/contract_sync_handler.py", line 42, in serialize_objs
     json.append(ContractFactory.map_server_to_mobile(obj))
 File "/crm/mobile_sync_system/factories/contract_factory.py", line 42, in map_server_to_mobile
     'fk_client': server_obj.client.mobile_uuid,
 File "/usr/local/lib/python3.6/site-packages/pony/orm/core.py", line 2288, in __get__
     value = attr.get(obj)
 File "/usr/local/lib/python3.6/site-packages/pony/orm/core.py", line 2298, in get
     val = vals[attr] if attr in vals else attr.load(obj)
 File "/usr/local/lib/python3.6/site-packages/pony/orm/core.py", line 2282, in load
     else: obj._load_()
 File "/usr/local/lib/python3.6/site-packages/pony/orm/core.py", line 4832, in _load_
     objects = entity._fetch_objects(cursor, attr_offsets)
 File "/usr/local/lib/python3.6/site-packages/pony/orm/core.py", line 4315, in _fetch_objects
     obj._db_set_(avdict)
 File "/usr/local/lib/python3.6/site-packages/pony/orm/core.py", line 4955, in _db_set_
     if attr.reverse: attr.db_update_reverse(obj, old_dbval, new_dbval)
 File "/usr/local/lib/python3.6/site-packages/pony/orm/core.py", line 2464, in db_update_reverse
     if new_dbval is not None: reverse.db_set(new_dbval, obj, True)
 File "/usr/local/lib/python3.6/site-packages/pony/orm/core.py", line 2419, in db_set
     assert old_val == old_dbval, (old_val, old_dbval)', 'AssertionError: (None, NOT_LOADED)

 During handling of the above exception, another exception occurred:', '
 Traceback (most recent call last):
 File "/crm/shared/logger/loggers.py", line 195, in _get_pretty_traceback
     log_data = cgitb.html(sys.exc_info())
 File "/usr/local/lib/python3.6/cgitb.py", line 136, in html
     vars = scanvars(reader, frame, locals)
 File "/usr/local/lib/python3.6/cgitb.py", line 88, in scanvars
     value = getattr(parent, token, __UNDEF__)
 File "/usr/local/lib/python3.6/site-packages/pony/orm/core.py", line 2288, in __get__
     value = attr.get(obj)
 File "/usr/local/lib/python3.6/site-packages/pony/orm/core.py", line 2298, in get
     val = vals[attr] if attr in vals else attr.load(obj)
 File "/usr/local/lib/python3.6/site-packages/pony/orm/core.py", line 2282, in load
      else: obj._load_()
 File "/usr/local/lib/python3.6/site-packages/pony/orm/core.py", line 4832, in _load_
     objects = entity._fetch_objects(cursor, attr_offsets)
 File "/usr/local/lib/python3.6/site-packages/pony/orm/core.py", line 4315, in _fetch_objects
     obj._db_set_(avdict)
 File "/usr/local/lib/python3.6/site-packages/pony/orm/core.py", line 4955, in _db_set_
     if attr.reverse: attr.db_update_reverse(obj, old_dbval, new_dbval)
 File "/usr/local/lib/python3.6/site-packages/pony/orm/core.py", line 2464, in db_update_reverse
     if new_dbval is not None: reverse.db_set(new_dbval, obj, True)
 File "/usr/local/lib/python3.6/site-packages/pony/orm/core.py", line 2419, in db_set
     assert old_val == old_dbval, (old_val, old_dbval)', 'AssertionError: (None, NOT_LOADED)
"""



from pony.orm import *

db = Database('sqlite', ':memory:')

class ReconciledPayments(db.Entity):
    id = PrimaryKey(int)
    foo = Optional(int)
    add_on_id = Optional("ContractAddOns")


class ContractAddOns(db.Entity):
    id = PrimaryKey(int)
    reconciled_payments = Set("ReconciledPayments")


db.generate_mapping(create_tables=True)

sql_debug(True)

with db_session:
    r1 = ReconciledPayments(id=1)
    r2 = ReconciledPayments(id=2)
    c1 = ContractAddOns(id=1)
    commit()
    rollback()

    r1 = ReconciledPayments._get_by_raw_pkval_((1,))
    r1.set(add_on_id=None, foo=123)

    r2 = ReconciledPayments[2]
    r2.set(add_on_id=None, foo=456)



if any(attr not in obj._vals_ for attr in avdict):
    obj._load_()

for attr, new_val in list(avdict.items()):
    assert attr in obj._vals_
    old_val = obj._vals_[attr]
    if new_val == old_val:
        avdict.pop(attr)





