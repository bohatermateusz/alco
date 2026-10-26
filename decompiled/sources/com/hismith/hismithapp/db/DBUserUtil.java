package com.hismith.hismithapp.db;

import android.content.ContentValues;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.model.UserModel;

/* loaded from: classes3.dex */
public class DBUserUtil {
    private static DatabaseHelper dbHepler;
    private static SQLiteDatabase dbRead;
    private static SQLiteDatabase dbWrite;

    private static void closeAll() {
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

    public static void insertDBMessageUserByUserDetails(UserModel userModel) {
        open();
        ContentValues contentValues = new ContentValues();
        contentValues.put("userId", Integer.valueOf(userModel.getUserId()));
        contentValues.put("king", Integer.valueOf(userModel.getKing()));
        contentValues.put("gradeIcon", userModel.getGradeIcon());
        contentValues.put("name", userModel.getName());
        contentValues.put("gender", Integer.valueOf(userModel.getGender()));
        contentValues.put("constellation", userModel.getConstellation());
        contentValues.put("headIcon", userModel.getHeadIcon());
        contentValues.put("alias", userModel.getAlias());
        dbWrite.replace(DatabaseHelper.TABLE_MESSAGE_USER, null, contentValues);
        closeAll();
    }

    private static void open() {
        DatabaseHelper databaseHelper = new DatabaseHelper(MyApplication.getApplication());
        dbHepler = databaseHelper;
        dbWrite = databaseHelper.getWritableDatabase();
        dbRead = dbHepler.getReadableDatabase();
    }

    public static UserModel selectUserById(int i) {
        UserModel userModel;
        open();
        Cursor query = dbRead.query(DatabaseHelper.TABLE_MESSAGE_USER, null, "userId = ?", new String[]{i + ""}, null, null, null);
        if (query == null || !query.moveToFirst()) {
            userModel = null;
        } else {
            userModel = new UserModel();
            userModel.setName(query.getString(query.getColumnIndexOrThrow("name")));
            userModel.setKing(query.getInt(query.getColumnIndexOrThrow("king")));
            if (query.getColumnIndex("gradeIcon") != -1) {
                userModel.setGradeIcon(query.getString(query.getColumnIndexOrThrow("gradeIcon")));
            }
            userModel.setHeadIcon(query.getString(query.getColumnIndexOrThrow("headIcon")));
            userModel.setGender(query.getInt(query.getColumnIndexOrThrow("gender")));
            userModel.setConstellation(query.getString(query.getColumnIndexOrThrow("constellation")));
            query.close();
        }
        closeAll();
        return userModel;
    }

    public static void updateDBReferenceUserName(int i, String str) {
        open();
        ContentValues contentValues = new ContentValues();
        contentValues.put("userName", str);
        dbWrite.update(DatabaseHelper.TABLE_MESSAGE_REFERENCE, contentValues, "userId = ?", new String[]{i + ""});
        closeAll();
    }

    public static void updateDBUserByUserAlias(int i, String str) {
        open();
        ContentValues contentValues = new ContentValues();
        contentValues.put("alias", str);
        dbWrite.update(DatabaseHelper.TABLE_MESSAGE_USER, contentValues, "userId = ?", new String[]{i + ""});
        closeAll();
    }
}
