import os
import aiosqlite
from pathlib import Path

async def db_init():
    connection = await aiosqlite.connect(
       os.path.join(
           Path(__file__).parent,
           'feedback_db.db'
       )
    )
    async with connection.cursor() as cursor:
        await cursor.execute('''
            CREATE TABLE IF NOT EXISTS Ratings (
            id INTEGER PRIMARY KEY,
            source_lng TEXT,
            target_lng TEXT,
            source_txt TEXT,
            translated_txt TEXT,
            page_lng TEXT,
            rating INTEGER
            )
        ''')
        await cursor.execute('''
            CREATE TABLE IF NOT EXISTS Improvements (
            id INTEGER PRIMARY KEY,
            source_lng TEXT,
            target_lng TEXT,
            source_txt TEXT,
            translated_txt_our TEXT,
            translated_txt_user TEXT,
            page_lng TEXT
            )
        ''')
        await connection.commit()
    return connection

async def write_rating(connection, request_data):
    async with connection.cursor() as cursor:
        request = (
            'INSERT INTO Ratings (source_lng, target_lng, source_txt,' 
            ' translated_txt, page_lng, rating)'
            ' VALUES (?, ?, ?, ?, ?, ?)'
        )
        await cursor.execute(
            request,
            (
                request_data['source_lng'],
                request_data['target_lng'],
                request_data['source_txt'],
                request_data['translated_txt'],
                request_data['page_lng'],
                request_data['rating']
            )
        )
        await connection.commit()
    return True

async def write_improvement(connection, request_data):
    async with connection.cursor() as cursor:
        request = (
            'INSERT INTO Improvements (source_lng, target_lng, source_txt,' 
            ' translated_txt_our, translated_txt_user, page_lng)'
            ' VALUES (?, ?, ?, ?, ?, ?)'
        )
        await cursor.execute(
            request,
            (
                request_data['source_lng'],
                request_data['target_lng'],
                request_data['source_txt'],
                request_data['translated_txt_our'],
                request_data['translated_txt_user'],
                request_data['page_lng']
            )
        )
        await connection.commit()
    return True
