from loguru import logger
from pathlib import Path

def setup_logger(name: str='eda'):
    """Configura un logger de `loguru` para salida dual: consola y archivo.

    Esta función establece un sistema de logging estandarizado para el proyecto.
    Crea un directorio 'logs' si no existe y configura dos "sinks" (salidas):
    1.  Consola: Muestra logs en tiempo real con colores para fácil lectura.
    2.  Archivo: Guarda los logs en un archivo rotativo en 'logs/{name}.log'.
        - Rotación: Se crea un nuevo archivo cuando el actual alcanza 5 MB.
        - Retención: Se conservan los últimos 10 archivos de log.
        - Compresión: Los archivos antiguos se comprimen en formato .zip.
        - Nivel: Solo se guardan los mensajes de nivel 'INFO' o superior.

    Args:
        name (str, optional): El nombre base para el archivo de log.
                              Defaults to "eda".

    Returns:
        Logger: El objeto logger de `loguru` configurado y listo para ser usado.

    Example:
        >>> logger = setup_logger("my_process")
        >>> logger.info("Este es un mensaje informativo.")
        >>> logger.warning("Esta es una advertencia.")
        >>> logger.error("Esto es un error crítico.")
    """
    Path('logs').mkdir(parents=True, exist_ok=True)
    logger.remove
    # consola
    logger.add(lambda msg: print(msg, end=''), colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")
    # archivo rotado
    logger.add("logs/{name}.log".format(name=name), rotation='5 MB', retention=10,
               compression='zip', level='INFO')
    return logger