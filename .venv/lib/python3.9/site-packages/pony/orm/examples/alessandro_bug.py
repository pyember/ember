from pony.orm import *

# from Api.exceptions import LimitPokemonException, ChatAlreadyActiveException

database = Database()


class User(database.Entity):
    user_id = PrimaryKey(str)
    # favorite_color = Set(int)

    owned_pokemons = Set("Pokemon", reverse="owner")

    is_admin = Required(bool, default=False)
    is_banned = Required(bool, default=False)

    @staticmethod
    @db_session
    def get_or_create(user_id: int) -> "User":
        return User.get(user_id=str(user_id)) or User(user_id=str(user_id))

    @staticmethod
    @db_session
    def get_by_id(user_id: int) -> "User":
        return User.get(user_id=str(user_id))

    @db_session
    def catch_pokemon(self, pokemon: "Pokemon"):
        if len(self.owned_pokemons) >= 6:
            raise LimitPokemonException("You can't have more than 6 pokemons")

        self.owned_pokemons.add(pokemon.caught_by(self))

    @db_session
    def remove_pokemon(self, pokemon: "Pokemon"):
        self.owned_pokemons.remove(pokemon)

    @db_session
    def set_favorite_color(self, color: tuple):
        self.favorite_color = color


class Pokemon(database.Entity):
    name = Required(str)
    pokemon_id = Required(int)
    sprite = Required(str)
    is_shiny = Required(bool)

    owner = Optional(User, reverse="owned_pokemons")

    spawned_chat_id = Optional(str)
    spawned_message_id = Optional(str)

    @property
    def captured(self) -> bool:
        return self.owner is not None

    @db_session
    def caught_by(self, user: "User"):
        self.spawned_chat_id = 'NULL'
        self.spawned_message_id = 'NULL'

        self.owner = user

    def __str__(self):
        return (self.name if not self.is_shiny else f"{self.name} <b>shiny</b>") + f" (ID: {self.pokemon_id})"


class Chat(database.Entity):
    chat_id = PrimaryKey(str)
    active = Required(bool, default=False)

    @db_session
    def activate(self):
        if self.active:
            raise ChatAlreadyActiveException("Chat is already active")

        self.active = True

    @db_session
    def deactivate(self):
        self.active = False

    @staticmethod
    @db_session
    def get_or_create(chat_id: int) -> "Chat":
        return Chat.get(chat_id=str(chat_id)) or Chat(chat_id=str(chat_id))

    @staticmethod
    @db_session
    def get_by_id(chat_id: int) -> "Chat":
        return Chat.get(chat_id=str(chat_id))


@db_session
def spawn_pokemon(chat_id: int, message_id: int, pokemon_json: dict, is_shiny: bool = False):
    Pokemon(
        name=pokemon_json["name"],
        pokemon_id=pokemon_json["id"],
        sprite=pokemon_json["sprites"]["front_shiny"] if is_shiny else pokemon_json["sprites"]["front_default"],
        is_shiny=is_shiny,
        chat_id=str(chat_id),
        message_id=str(message_id)
    )


@db_session
def get_spawned_pokemon(chat_id: int, message_id: int) -> Pokemon:
    return Pokemon.get(chat_id=str(chat_id), message_id=str(message_id))


def setup():
    database.bind(provider='sqlite', filename=':memory:')
    database.generate_mapping(create_tables=True)

setup()
