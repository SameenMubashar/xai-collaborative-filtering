"""
Data Loader for MovieLens Dataset
Downloads, cleans, and prepares the MovieLens 1M dataset for collaborative filtering
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path


class MovieLensDataLoader:
    """
    Handles downloading and preprocessing of MovieLens datasets
    """
    
    def __init__(self, dataset_size='1m', data_dir='data/raw'):
        """
        Initialize the data loader
        
        Args:
            dataset_size: '100k', '1m', '10m', or '25m'
            data_dir: Directory to store downloaded data
        """
        self.dataset_size = dataset_size
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs
        self.urls = {
            '100k': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
            '1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
            '10m': 'https://files.grouplens.org/datasets/movielens/ml-10m.zip',
            '25m': 'https://files.grouplens.org/datasets/movielens/ml-25m.zip'
        }
        
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        self.merged_df = None
    
    def download_dataset(self):
        """
        Download the dataset if it doesn't exist
        """
        url = self.urls.get(self.dataset_size)
        if not url:
            raise ValueError(f"Invalid dataset size: {self.dataset_size}")
        
        zip_path = self.data_dir / f'ml-{self.dataset_size}.zip'
        extract_dir = self.data_dir / f'ml-{self.dataset_size}'
        
        # Check if already downloaded
        if extract_dir.exists():
            print(f"✓ Dataset ml-{self.dataset_size} already exists at {extract_dir}")
            return extract_dir
        
        print(f"Downloading MovieLens {self.dataset_size.upper()} dataset...")
        print(f"URL: {url}")
        
        # Download with progress
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rProgress: {percent}%", end='')
        
        urllib.request.urlretrieve(url, zip_path, reporthook=progress_hook)
        print(f"\n✓ Downloaded to {zip_path}")
        
        # Extract
        print(f"Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        print(f"✓ Extracted to {extract_dir}")
        
        # Remove zip file to save space
        zip_path.unlink()
        print(f"✓ Cleaned up zip file")
        
        return extract_dir
    
    def load_ratings(self):
        """
        Load and clean ratings data
        """
        extract_dir = self.download_dataset()
        
        if self.dataset_size == '1m':
            # MovieLens 1M uses :: as delimiter
            ratings_file = extract_dir / 'ratings.dat'
            print(f"\nLoading ratings from {ratings_file}...")
            
            self.ratings_df = pd.read_csv(
                ratings_file,
                sep='::',
                engine='python',
                names=['user_id', 'movie_id', 'rating', 'timestamp'],
                encoding='latin-1'
            )
        elif self.dataset_size == '100k':
            # MovieLens 100K uses tabs
            ratings_file = extract_dir / 'u.data'
            self.ratings_df = pd.read_csv(
                ratings_file,
                sep='\t',
                names=['user_id', 'movie_id', 'rating', 'timestamp']
            )
        else:
            # 10M and 25M use comma delimiter
            ratings_file = extract_dir / 'ratings.csv'
            self.ratings_df = pd.read_csv(ratings_file)
            if 'userId' in self.ratings_df.columns:
                self.ratings_df.rename(columns={
                    'userId': 'user_id',
                    'movieId': 'movie_id'
                }, inplace=True)
        
        # Convert timestamp to datetime
        self.ratings_df['timestamp'] = pd.to_datetime(
            self.ratings_df['timestamp'], 
            unit='s'
        )
        
        print(f"✓ Loaded {len(self.ratings_df):,} ratings")
        return self.ratings_df
    
    def load_movies(self):
        """
        Load and clean movies data
        """
        extract_dir = self.data_dir / f'ml-{self.dataset_size}'
        
        if self.dataset_size == '1m':
            movies_file = extract_dir / 'movies.dat'
            print(f"Loading movies from {movies_file}...")
            
            self.movies_df = pd.read_csv(
                movies_file,
                sep='::',
                engine='python',
                names=['movie_id', 'title', 'genres'],
                encoding='latin-1'
            )
        elif self.dataset_size == '100k':
            movies_file = extract_dir / 'u.item'
            self.movies_df = pd.read_csv(
                movies_file,
                sep='|',
                encoding='latin-1',
                names=['movie_id', 'title', 'release_date', 'video_release_date',
                       'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                       'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                       'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                       'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
            )
            # Combine genre columns into single genres column
            genre_cols = self.movies_df.columns[6:]
            self.movies_df['genres'] = self.movies_df[genre_cols].apply(
                lambda row: '|'.join([col for col, val in row.items() if val == 1]),
                axis=1
            )
            self.movies_df = self.movies_df[['movie_id', 'title', 'genres']]
        else:
            movies_file = extract_dir / 'movies.csv'
            self.movies_df = pd.read_csv(movies_file)
            if 'movieId' in self.movies_df.columns:
                self.movies_df.rename(columns={'movieId': 'movie_id'}, inplace=True)
        
        # Extract year from title
        self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)')
        self.movies_df['year'] = pd.to_numeric(self.movies_df['year'], errors='coerce')
        
        # Split genres into list
        self.movies_df['genre_list'] = self.movies_df['genres'].str.split('|')
        
        print(f"✓ Loaded {len(self.movies_df):,} movies")
        return self.movies_df
    
    def load_users(self):
        """
        Load and clean users data (only available for 1M dataset)
        """
        if self.dataset_size != '1m':
            print(f"User demographic data not available for {self.dataset_size} dataset")
            return None
        
        extract_dir = self.data_dir / f'ml-{self.dataset_size}'
        users_file = extract_dir / 'users.dat'
        
        print(f"Loading users from {users_file}...")
        
        self.users_df = pd.read_csv(
            users_file,
            sep='::',
            engine='python',
            names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
            encoding='latin-1'
        )
        
        # Map age codes to age ranges (for 1M dataset)
        age_mapping = {
            1: "Under 18",
            18: "18-24",
            25: "25-34",
            35: "35-44",
            45: "45-49",
            50: "50-55",
            56: "56+"
        }
        self.users_df['age_group'] = self.users_df['age'].map(age_mapping)
        
        # Map occupation codes
        occupation_mapping = {
            0: "other", 1: "academic/educator", 2: "artist",
            3: "clerical/admin", 4: "college/grad student",
            5: "customer service", 6: "doctor/health care",
            7: "executive/managerial", 8: "farmer", 9: "homemaker",
            10: "K-12 student", 11: "lawyer", 12: "programmer",
            13: "retired", 14: "sales/marketing", 15: "scientist",
            16: "self-employed", 17: "technician/engineer",
            18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
        }
        self.users_df['occupation_name'] = self.users_df['occupation'].map(occupation_mapping)
        
        print(f"✓ Loaded {len(self.users_df):,} users")
        return self.users_df
    
    def create_merged_dataset(self):
        """
        Merge ratings, movies, and users data
        """
        if self.ratings_df is None:
            self.load_ratings()
        if self.movies_df is None:
            self.load_movies()
        if self.dataset_size == '1m' and self.users_df is None:
            self.load_users()
        
        print("\nMerging datasets...")
        
        # Merge ratings with movies
        self.merged_df = self.ratings_df.merge(
            self.movies_df,
            on='movie_id',
            how='left'
        )
        
        # Merge with users if available
        if self.users_df is not None:
            self.merged_df = self.merged_df.merge(
                self.users_df,
                on='user_id',
                how='left'
            )
        
        print(f"✓ Created merged dataset with {len(self.merged_df):,} records")
        return self.merged_df
    
    def get_statistics(self):
        """
        Get dataset statistics
        """
        if self.merged_df is None:
            self.create_merged_dataset()
        
        stats = {
            'total_ratings': len(self.merged_df),
            'unique_users': self.merged_df['user_id'].nunique(),
            'unique_movies': self.merged_df['movie_id'].nunique(),
            'rating_range': (self.merged_df['rating'].min(), self.merged_df['rating'].max()),
            'avg_rating': self.merged_df['rating'].mean(),
            'sparsity': 1 - (len(self.merged_df) / (
                self.merged_df['user_id'].nunique() * 
                self.merged_df['movie_id'].nunique()
            )),
            'date_range': (
                self.merged_df['timestamp'].min(),
                self.merged_df['timestamp'].max()
            )
        }
        
        return stats
    
    def print_statistics(self):
        """
        Print dataset statistics
        """
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print(f"MovieLens {self.dataset_size.upper()} Dataset Statistics")
        print("="*60)
        print(f"Total Ratings:       {stats['total_ratings']:,}")
        print(f"Unique Users:        {stats['unique_users']:,}")
        print(f"Unique Movies:       {stats['unique_movies']:,}")
        print(f"Rating Range:        {stats['rating_range'][0]} - {stats['rating_range'][1]}")
        print(f"Average Rating:      {stats['avg_rating']:.2f}")
        print(f"Sparsity:            {stats['sparsity']*100:.2f}%")
        print(f"Date Range:          {stats['date_range'][0].date()} to {stats['date_range'][1].date()}")
        print(f"Ratings per User:    {stats['total_ratings'] / stats['unique_users']:.1f}")
        print(f"Ratings per Movie:   {stats['total_ratings'] / stats['unique_movies']:.1f}")
        print("="*60)
    
    def save_processed_data(self, output_dir='data/processed'):
        """
        Save processed data to CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.ratings_df is not None:
            ratings_file = output_path / f'ratings_{self.dataset_size}.csv'
            self.ratings_df.to_csv(ratings_file, index=False)
            print(f"✓ Saved ratings to {ratings_file}")
        
        if self.movies_df is not None:
            movies_file = output_path / f'movies_{self.dataset_size}.csv'
            self.movies_df.to_csv(movies_file, index=False)
            print(f"✓ Saved movies to {movies_file}")
        
        if self.users_df is not None:
            users_file = output_path / f'users_{self.dataset_size}.csv'
            self.users_df.to_csv(users_file, index=False)
            print(f"✓ Saved users to {users_file}")
        
        if self.merged_df is not None:
            merged_file = output_path / f'merged_{self.dataset_size}.csv'
            self.merged_df.to_csv(merged_file, index=False)
            print(f"✓ Saved merged dataset to {merged_file}")


def main():
    """
    Main function to demonstrate data loading
    """
    # Load MovieLens 1M dataset
    loader = MovieLensDataLoader(dataset_size='1m')
    
    # Load all data
    print("="*60)
    print("Loading MovieLens 1M Dataset")
    print("="*60)
    
    loader.load_ratings()
    loader.load_movies()
    loader.load_users()
    loader.create_merged_dataset()
    
    # Print statistics
    loader.print_statistics()
    
    # Show sample data
    print("\n" + "="*60)
    print("Sample Data")
    print("="*60)
    print("\nRatings Sample:")
    print(loader.ratings_df.head())
    print("\nMovies Sample:")
    print(loader.movies_df.head())
    if loader.users_df is not None:
        print("\nUsers Sample:")
        print(loader.users_df.head())
    print("\nMerged Dataset Sample:")
    print(loader.merged_df.head())
    
    # Save processed data
    print("\n" + "="*60)
    print("Saving Processed Data")
    print("="*60)
    loader.save_processed_data()
    
    print("\n✓ Data loading and preprocessing complete!")


if __name__ == "__main__":
    main()
