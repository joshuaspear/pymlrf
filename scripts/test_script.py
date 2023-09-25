import argparse
import logging
# import classes/functions etc from main library

logger = logging.getLogger("<name_of_logger>")

def main(logdir:str):
    """Main function

    Args:
        logdir (str): Directory to write logging files to
    """
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument("--logdir", type=str, default="/", required=False)
    
    # Parse the argument
    args = parser.parse_args()

    main(args.logdir)