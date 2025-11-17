#!/usr/bin/env python
"""
Test script for list_benzinga_news v2 API integration.

This script tests the updated list_benzinga_news function to verify it works
with the v2 API endpoint.

Usage:
    # Option 1: Using .env file (recommended)
    # Create a .env file with: MASSIVE_API_KEY=your_api_key_here
    uv run python -m tests.test_benzinga_news
    
    # Option 2: Using environment variable
    MASSIVE_API_KEY=your_api_key_here uv run python -m tests.test_benzinga_news
"""
import os
import asyncio
from dotenv import load_dotenv
from mcp_massive.server import list_benzinga_news

# Load .env file if it exists
load_dotenv()


async def test_basic_query():
    """Test basic query with default parameters."""
    print("=" * 60)
    print("Test 1: Basic query with default parameters")
    print("=" * 60)
    try:
        result = await list_benzinga_news(limit=5)
        print(f"✓ Success! Returned {len(result.splitlines()) - 1} rows (excluding header)")
        print(f"First 500 characters of result:\n{result[:500]}...")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


async def test_with_ticker_filter():
    """Test query with ticker filter."""
    print("\n" + "=" * 60)
    print("Test 2: Query with ticker filter (AAPL)")
    print("=" * 60)
    try:
        result = await list_benzinga_news(tickers="AAPL", limit=3)
        print(f"✓ Success! Returned {len(result.splitlines()) - 1} rows")
        print(f"First 500 characters of result:\n{result[:500]}...")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


async def test_with_published_date():
    """Test query with published date filter."""
    print("\n" + "=" * 60)
    print("Test 3: Query with published date filter (today)")
    print("=" * 60)
    from datetime import date
    today = date.today().isoformat()
    try:
        result = await list_benzinga_news(published=today, limit=3)
        row_count = len(result.splitlines()) - 1 if result.strip() else 0
        if row_count > 0:
            print(f"✓ Success! Returned {row_count} rows")
            print(f"First 500 characters of result:\n{result[:500]}...")
        else:
            print(f"⚠ No results for today ({today}). This is normal if there are no articles published today.")
            print("The API call succeeded, but returned no matching articles.")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


async def test_with_channels():
    """Test query with channels filter."""
    print("\n" + "=" * 60)
    print("Test 4: Query with channels filter")
    print("=" * 60)
    try:
        result = await list_benzinga_news(channels="News", limit=3)
        row_count = len(result.splitlines()) - 1 if result.strip() else 0
        if row_count > 0:
            print(f"✓ Success! Returned {row_count} rows")
            print(f"First 500 characters of result:\n{result[:500]}...")
        else:
            print(f"⚠ No results for channel 'News'. This might mean:")
            print("  - The channel name doesn't match exactly")
            print("  - There are no articles in that channel currently")
            print("  - Try checking available channels from a basic query result")
            print("The API call succeeded, but returned no matching articles.")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


async def test_with_sort():
    """Test query with sort parameter."""
    print("\n" + "=" * 60)
    print("Test 5: Query with sort parameter")
    print("=" * 60)
    try:
        result = await list_benzinga_news(sort="published.desc", limit=3)
        print(f"✓ Success! Returned {len(result.splitlines()) - 1} rows")
        print(f"First 500 characters of result:\n{result[:500]}...")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


async def test_multiple_filters():
    """Test query with multiple filters."""
    print("\n" + "=" * 60)
    print("Test 6: Query with multiple filters (ticker)")
    print("=" * 60)
    try:
        result = await list_benzinga_news(
            tickers="TSLA",
            limit=2
        )
        print(f"✓ Success! Returned {len(result.splitlines()) - 1} rows")
        print(f"First 500 characters of result:\n{result[:500]}...")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


async def main():
    """Run all tests."""
    # Check for API key
    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        print("ERROR: MASSIVE_API_KEY environment variable not set!")
        print("Please set it before running tests:")
        print("  Option 1: Create a .env file with: MASSIVE_API_KEY=your_api_key_here")
        print("  Option 2: Export it: export MASSIVE_API_KEY=your_api_key_here")
        return
    
    print("Testing list_benzinga_news v2 API integration")
    print(f"API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '****'}")
    print()
    
    tests = [
        test_basic_query,
        test_with_ticker_filter,
        test_with_published_date,
        test_with_channels,
        test_with_sort,
        test_multiple_filters,
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
        # Small delay between tests to avoid rate limiting
        await asyncio.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

