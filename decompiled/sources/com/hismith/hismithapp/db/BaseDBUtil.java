package com.hismith.hismithapp.db;

import android.content.ContentValues;
import android.database.sqlite.SQLiteDatabase;
import com.hismith.hismithapp.application.MyApplication;

/* loaded from: classes3.dex */
public class BaseDBUtil {
    protected static DatabaseHelper dbHepler;
    protected static SQLiteDatabase dbRead;
    protected static SQLiteDatabase dbWrite;

    protected static void closeAll() {
        SQLiteDatabase sQLiteDatabase = dbRead;
        if (sQLiteDatabase != null) {
            sQLiteDatabase.close();
        }
        SQLiteDatabase sQLiteDatabase2 = dbWrite;
        if (sQLiteDatabase2 != null) {
            sQLiteDatabase2.close();
        }
        DatabaseHelper databaseHelper = dbHepler;
        if (databaseHelper != null) {
            databaseHelper.close();
        }
    }

    protected static void deleteData(String str, String str2, String[] strArr) {
        open();
        dbWrite.delete(str, str2, strArr);
        closeAll();
    }

    protected static void insertData(String str, ContentValues contentValues) {
        open();
        dbWrite.replace(str, null, contentValues);
        closeAll();
    }

    protected static void open() {
        DatabaseHelper databaseHelper = new DatabaseHelper(MyApplication.getApplication());
        dbHepler = databaseHelper;
        dbWrite = databaseHelper.getWritableDatabase();
        dbRead = dbHepler.getReadableDatabase();
    }
}
