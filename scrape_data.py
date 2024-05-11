# Cargar solo las columnas "Fecha" y "Valor" del archivo "temp_max.csv"
max_temp = pd.read_csv("temp_max.csv", usecols=["Fecha", "Valor"], parse_dates=["Fecha"], date_parser=lambda x: pd.to_datetime(x).date())

# Cargar solo las columnas "Fecha" y "Valor" del archivo "temp_min.csv"
min_temp = pd.read_csv("temp_min.csv", usecols=["Fecha", "Valor"], parse_dates=["Fecha"], date_parser=lambda x: pd.to_datetime(x).date())

# Cambiar el nombre de la columna "Valor" a "Tmax" para max_temp
max_temp.rename(columns={'Valor': 'Tmax'}, inplace=True)

# Cambiar el nombre de la columna "Valor" a "Tmin" para min_temp
min_temp.rename(columns={'Valor': 'Tmin'}, inplace=True)

# Cambiar el nombre de la columna "Valor" a "Tmax" para max_temp
max_temp.rename(columns={'Fecha': 'Date'}, inplace=True)

# Cambiar el nombre de la columna "Valor" a "Tmin" para min_temp
min_temp.rename(columns={'Fecha': 'Date'}, inplace=True)


# Establecer la columna "Fecha" como índice para max_temp y min_temp
max_temp.set_index('Date', inplace=True)
min_temp.set_index('Date', inplace=True)