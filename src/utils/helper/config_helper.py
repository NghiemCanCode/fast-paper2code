from file_helper import load_yaml


class ConfigParser:
    def __init__(self, args ,config_file_path):
        self._config_data = {}
        self._config_file_file_path = config_file_path

        self._config_data = load_yaml(self._config_file_file_path)

        for key, value in vars(args).items():
            if value:
                self._config_data[key] = value

    @property
    def config_data(self):
        return self._config_data