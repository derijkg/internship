# main script to run all
import argparse
from scrape import ScriptiebankScraper
import pandas as pd
import re



def main():
    #path_df = Path('data2/')


    #parser = argparse.ArgumentParser(description="A simple script to demonstrate argparse.")
    #parser.add_argument("input_file", type=str, help="Path to the input file.")
    #parser.add_argument("-o", "--output", type=str, default="output.txt", help="Path to the output file (default: output.txt).")
    #parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    #args = parser.parse_args()

    #print(f"Input file: {args.input_file}")
    #print(f"Output file: {args.output}")

    #if args.verbose:
    #    print("Verbose mode enabled.")

    scraper = ScriptiebankScraper()
    scraper.run(gather_metadata=True,gather_urls=True,download_files=True)





    #generate more robust id: source_name+download_link_token
    #df = pd.read_csv()
    #download_pattern = re.compile(r"token=([a-zA-Z0-9_-]+)")
    #df['download_token'] = df['download_link'].str.extract(download_pattern)[0]
    #df['id'] = df['source'] + '_' + df['download_token']

    # Display the updated DataFrame
    #print(df)




    # json -> tsv from ugent
    




if __name__ == "__main__":
    main()

