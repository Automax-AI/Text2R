# Automax AI Market Conditions Tool
## Prerequisites

- Python 3.9+
- R 4.0+ installed on your system
- Supabase account for database functionality
- OpenAI API key

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/automax-market-tool.git
   cd automax-market-tool
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

  create a .env file in the root directory and add the following:
  ```
  OPENAI_API_KEY=your-openai-api-key
  SUPABASE_URL=your-supabase-url
  SUPABASE_KEY=your-supabase-key
  ```

5. **Run the application**
   ```bash
   streamlit run Rpy2.py
   ```

## Usage

### Account Creation and Login

1. Register a new account or login to an existing account
2. Your data and chat history will be saved to your account

### Data Upload and Management

1. Upload CSV or Excel files containing your market data
2. Preview your data before analysis
3. Clear history or start new conversations as needed

### Creating Visualizations

1. Enter a request describing the visualization you want
2. The AI will generate appropriate R code and explanations
3. View the generated visualization and the explanation
4. Your conversation history is saved for future reference

## Deployment

For information on deploying the application to production environments, see the [DEPLOYMENT.md](DEPLOYMENT.md) guide.

## Technologies Used

- **Streamlit**: For the web interface
- **OpenAI**: For AI-powered analysis
- **rpy2**: For Python-R integration
- **Supabase**: For cloud database functionality and user management
- **Pandas**: For data processing
- **R/ggplot2**: For statistical visualizations

## Contributing

Contributions to improve the application are welcome. Please feel free to submit a Pull Request.

## Acknowledgments

- This project uses various open-source libraries and tools.
- Special thanks to the R community for the excellent visualization capabilities.

## Supabase Setup Guide

### Creating Required Tables

The application requires two tables in your Supabase database. You can create them using the SQL Editor in the Supabase Dashboard:

```sql
-- User files table
CREATE TABLE public.user_files (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL,
  filename TEXT NOT NULL,
  file_data TEXT NOT NULL,
  content_type TEXT NOT NULL,
  uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Chat history table
CREATE TABLE public.chat_history (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL,
  messages JSONB NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.user_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_history ENABLE ROW LEVEL SECURITY;

-- Create policies for user_files table
CREATE POLICY "Allow users access to own files" ON public.user_files
  FOR ALL USING (auth.uid() = user_id);

-- Create policies for chat_history table
CREATE POLICY "Allow users access to own chat history" ON public.chat_history
  FOR ALL USING (auth.uid() = user_id);
```

Note: You don't need to create a custom users table because Supabase Auth automatically manages user authentication and stores user data in its own internal tables.

### Authentication Setup

1. Go to the Authentication section in your Supabase Dashboard
2. Under Email Auth, make sure Email provider is enabled
3. Configure email templates if needed
4. Optionally, customize the Site URL to match your deployed application

### Environment Variables

Make sure to set these in your `.env` file:

```
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-key
OPENAI_API_KEY=your-openai-api-key
```

You can find the Supabase URL and anon key in your Supabase project settings under "API".
