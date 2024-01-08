import sys
from sqlalchemy import create_engine
from sqlalchemy.engine import reflection

def get_table_names(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    inspector = reflection.Inspector.from_engine(engine)
    return inspector.get_table_names()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        database_filepath = sys.argv[1]
        tables = get_table_names(database_filepath)
        print('Tables in the database:')
        for table in tables:
            print(table)
    else:
        print('Please provide the filepath of the disaster messages database as the argument.')
        print('Example: python list_tables.py data/DisasterResponse.db')