import aiohttp
import asyncio
from tqdm.asyncio import tqdm_asyncio


from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger



# function to retrieve HTML data
###############################################################################
async def data_from_single_URL(session, url, semaphore):
    async with semaphore:
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f'Could not fetch data from {url}. Status: {response.status}')
                return None
            try:
                return await response.json()
            except aiohttp.client_exceptions.ContentTypeError as e:
                logger.error(f'Error decoding JSON from {url}: {e}')
                return None   
            

# function to retrieve HTML data
###############################################################################
async def data_from_multiple_URLs(urls, num_calls):
    semaphore = asyncio.Semaphore(num_calls)
    async with aiohttp.ClientSession() as session:
        tasks = [data_from_single_URL(session, url, semaphore) for url in urls]
        results = await tqdm_asyncio.gather(*tasks)
    return results


