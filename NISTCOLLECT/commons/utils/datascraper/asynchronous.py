import aiohttp
import asyncio
import pubchempy as pcp
from tqdm.asyncio import tqdm_asyncio

from NISTCOLLECT.commons.constants import CONFIG
from NISTCOLLECT.commons.logger import logger



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
async def properties_from_single_name(name, semaphore):
    async with semaphore:
        try:            
            cid = pcp.get_cids(name, list_return='flat')
            properties = pcp.Compound.from_cid(cid).to_dict()
            logger.debug(f'Successfully retrieved properties for {name}')
        except IndexError:
            logger.error(f'No CID found for {name}')
            properties = {}
        except Exception as e:
            logger.error(f'Error fetching properties for {name}: {e}')
            properties = {}
            
        return properties
            

# function to retrieve HTML data
###############################################################################
async def data_from_multiple_URLs(urls):
    semaphore = asyncio.Semaphore(CONFIG["PARALLEL_TASKS"])
    async with aiohttp.ClientSession() as session:
        tasks = [data_from_single_URL(session, url, semaphore) for url in urls]
        results = await tqdm_asyncio.gather(*tasks)
    return results


# function to retrieve HTML data
###############################################################################
async def properties_from_multiple_names(names):
    semaphore = asyncio.Semaphore(CONFIG["PARALLEL_TASKS"])    
    tasks = [properties_from_single_name(name, semaphore) for name in names]
    results = await tqdm_asyncio.gather(*tasks)

    return results

