package com.hismith.hismithapp.db;

import android.content.ContentValues;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.model.UserModel;

/* loaded from: classes3.dex */
public class DBUtil {
    private DatabaseHelper dbHepler;
    private SQLiteDatabase dbRead;
    private SQLiteDatabase dbWrite;

    class DBUserModel {
        private int age;
        private String constellation;
        private int gender;
        private String gradeIcon;
        private String headIcon;
        private int king;
        private String name;
        private int userId;

        DBUserModel() {
        }

        public int getAge() {
            return this.age;
        }

        public String getConstellation() {
            return this.constellation;
        }

        public int getGender() {
            return this.gender;
        }

        public String getGradeIcon() {
            return this.gradeIcon;
        }

        public String getHeadIcon() {
            return this.headIcon;
        }

        public int getKing() {
            return this.king;
        }

        public String getName() {
            return this.name;
        }

        public int getUserId() {
            return this.userId;
        }

        public void setAge(int i) {
            this.age = i;
        }

        public void setConstellation(String str) {
            this.constellation = str;
        }

        public void setGender(int i) {
            this.gender = i;
        }

        public void setGradeIcon(String str) {
            this.gradeIcon = str;
        }

        public void setHeadIcon(String str) {
            this.headIcon = str;
        }

        public void setKing(int i) {
            this.king = i;
        }

        public void setName(String str) {
            this.name = str;
        }

        public void setUserId(int i) {
            this.userId = i;
        }
    }

    public DBUtil() {
        DatabaseHelper databaseHelper = new DatabaseHelper(MyApplication.getApplication());
        this.dbHepler = databaseHelper;
        this.dbWrite = databaseHelper.getWritableDatabase();
        this.dbRead = this.dbHepler.getReadableDatabase();
    }

    private void closeAll() {
        SQLiteDatabase sQLiteDatabase = this.dbRead;
        if (sQLiteDatabase != null) {
            sQLiteDatabase.close();
        }
        SQLiteDatabase sQLiteDatabase2 = this.dbWrite;
        if (sQLiteDatabase2 != null) {
            sQLiteDatabase2.close();
        }
        DatabaseHelper databaseHelper = this.dbHepler;
        if (databaseHelper != null) {
            databaseHelper.close();
        }
    }

    private void insertDBUser(UserModel userModel) {
        ContentValues contentValues = new ContentValues();
        contentValues.put("userId", Integer.valueOf(userModel.getUserId()));
        contentValues.put("king", Integer.valueOf(userModel.getKing()));
        contentValues.put("gradeIcon", userModel.getGradeIcon());
        contentValues.put("name", userModel.getName());
        contentValues.put("gender", Integer.valueOf(userModel.getGender()));
        contentValues.put("constellation", userModel.getConstellation());
        contentValues.put("headIcon", userModel.getHeadIcon());
        this.dbWrite.replace(DatabaseHelper.TABLE_MESSAGE_USER, null, contentValues);
    }

    private void readDBUser(int i) {
        DBUserModel dBUserModel = new DBUserModel();
        Cursor query = this.dbRead.query(DatabaseHelper.TABLE_MESSAGE_USER, null, "userId = ?", new String[]{i + ""}, null, null, null);
        if (query == null || !query.moveToFirst()) {
            return;
        }
        dBUserModel.setUserId(query.getInt(query.getColumnIndex("userId")));
        dBUserModel.setName(query.getString(query.getColumnIndex("name")));
        dBUserModel.setKing(query.getInt(query.getColumnIndex("king")));
        if (query.getColumnIndex("gradeIcon") != -1) {
            dBUserModel.setGradeIcon(query.getString(query.getColumnIndex("gradeIcon")));
        }
        dBUserModel.setAge(query.getInt(query.getColumnIndex("age")));
        dBUserModel.setHeadIcon(query.getString(query.getColumnIndex("headIcon")));
        dBUserModel.setGender(query.getInt(query.getColumnIndex("gender")));
        dBUserModel.setConstellation(query.getString(query.getColumnIndex("constellation")));
        query.close();
    }
}
