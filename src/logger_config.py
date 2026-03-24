import logging
import datetime
import os

def setup_logging():
    # Création du dossier logs
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Nom unique basé sur le lancement
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/session_{timestamp}.log"

    # Configuration globale
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("MonApp")

# On initialise le logger principal ici
logger = setup_logging()