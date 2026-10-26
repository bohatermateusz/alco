package com.hismith.hismithapp.db;

import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import com.hismith.hismithapp.application.MyApplication;

/* loaded from: classes3.dex */
public class DBMessageLastUtil {
    private static DatabaseHelper dbHepler;
    private static SQLiteDatabase dbRead;

    private static void closeAll() {
        SQLiteDatabase sQLiteDatabase = dbRead;
        if (sQLiteDatabase != null) {
            sQLiteDatabase.close();
        }
        DatabaseHelper databaseHelper = dbHepler;
        if (databaseHelper != null) {
            databaseHelper.close();
        }
    }

    public static int getMessageLastId() {
        int i = 0;
        try {
            open();
            Cursor query = dbRead.query(DatabaseHelper.TABLE_MESSAGE_LAST, new String[]{"msgLastId"}, "sendUserId = ?", new String[]{MyApplication.getApplication().getmUserModel().getUserId() + ""}, null, null, null);
            if (query != null && query.moveToFirst()) {
                do {
                    i = query.getInt(query.getColumnIndex("msgLastId"));
                } while (query.moveToNext());
            }
            if (query != null) {
                query.close();
            }
            closeAll();
            return i;
        } catch (Exception e) {
            e.printStackTrace();
            return i;
        }
    }

    private static void open() {
        DatabaseHelper databaseHelper = new DatabaseHelper(MyApplication.getApplication());
        dbHepler = databaseHelper;
        dbRead = databaseHelper.getReadableDatabase();
    }
}
