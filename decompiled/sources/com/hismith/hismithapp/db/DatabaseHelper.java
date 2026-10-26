package com.hismith.hismithapp.db;

import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.text.TextUtils;
import com.hismith.hismithapp.BuildConfig;
import com.hismith.hismithapp.utils.LogUtil;

/* loaded from: classes3.dex */
public class DatabaseHelper extends SQLiteOpenHelper {
    public static String TABLE_MESSAGE = "message";
    public static String TABLE_MESSAGE_CONVERSATION = "message_conversation";
    public static String TABLE_MESSAGE_LAST = "message_last";
    public static String TABLE_MESSAGE_REFERENCE = "message_reference";
    public static String TABLE_MESSAGE_USER = "message_user";
    public static String TABLE_TOPIC_DRAFT = "topic_draft";

    public DatabaseHelper(Context context) {
        super(context, BuildConfig.DB_NAME, (SQLiteDatabase.CursorFactory) null, 13);
    }

    private void addColumn(SQLiteDatabase sQLiteDatabase, String[] strArr, String str) {
        Cursor rawQuery;
        if (sQLiteDatabase == null || strArr == null || strArr.length < 1 || TextUtils.isEmpty(str) || (rawQuery = sQLiteDatabase.rawQuery("select * from " + str, null)) == null) {
            return;
        }
        String[] columnNames = rawQuery.getColumnNames();
        String str2 = "_temp_" + str;
        sQLiteDatabase.execSQL("alter table " + str + " rename to " + str2);
        if (columnNames.length < 1) {
            return;
        }
        StringBuffer stringBuffer = new StringBuffer();
        stringBuffer.append("create table if not exists " + str + "(");
        for (int i = 0; i < columnNames.length; i++) {
            if (i == 0) {
                stringBuffer.append(columnNames[i] + " integer primary key autoincrement,");
            } else {
                stringBuffer.append(columnNames[i] + ",");
            }
        }
        for (int i2 = 0; i2 < strArr.length; i2++) {
            if (i2 == strArr.length - 1) {
                stringBuffer.append(strArr[i2] + ")");
            } else {
                stringBuffer.append(strArr[i2] + ",");
            }
        }
        sQLiteDatabase.execSQL(stringBuffer.toString());
        StringBuffer stringBuffer2 = new StringBuffer();
        stringBuffer2.append("insert into " + str + " select *,");
        for (int i3 = 0; i3 < strArr.length; i3++) {
            if (i3 == strArr.length - 1) {
                stringBuffer2.append("' ' from " + str2);
            } else {
                stringBuffer2.append("' ',");
            }
        }
        sQLiteDatabase.execSQL(stringBuffer2.toString());
        sQLiteDatabase.execSQL("drop table " + str2);
        rawQuery.close();
    }

    private void upgradeToVersion07(SQLiteDatabase sQLiteDatabase) {
        sQLiteDatabase.execSQL("CREATE TABLE " + TABLE_MESSAGE_LAST + " (id integer primary key autoincrement, sendUserId integer unique, msgLastId integer)");
    }

    private void upgradeToVersion08(SQLiteDatabase sQLiteDatabase) {
        sQLiteDatabase.execSQL("ALTER TABLE " + TABLE_MESSAGE + " ADD COLUMN localPath text");
    }

    private void upgradeToVersion09(SQLiteDatabase sQLiteDatabase) {
        sQLiteDatabase.execSQL("ALTER TABLE " + TABLE_MESSAGE + " ADD COLUMN reject integer");
    }

    private void upgradeToVersion10(SQLiteDatabase sQLiteDatabase) {
        sQLiteDatabase.execSQL("ALTER TABLE " + TABLE_MESSAGE_USER + " ADD COLUMN gradeIcon text");
    }

    private void upgradeToVersion11(SQLiteDatabase sQLiteDatabase) {
        sQLiteDatabase.execSQL("CREATE TABLE " + TABLE_TOPIC_DRAFT + " (id integer primary key autoincrement, userId integer , title text , content text , contentImage text , contentText text , date text ,  categorieId text)");
    }

    private void upgradeToVersion12(SQLiteDatabase sQLiteDatabase) {
        sQLiteDatabase.execSQL("ALTER TABLE " + TABLE_MESSAGE + " ADD COLUMN alias text");
    }

    private void upgradeToVersion13(SQLiteDatabase sQLiteDatabase) {
        sQLiteDatabase.execSQL("ALTER TABLE " + TABLE_MESSAGE + " ADD COLUMN referenceId integer");
        sQLiteDatabase.execSQL("CREATE TABLE " + TABLE_MESSAGE_REFERENCE + " (id integer primary key autoincrement, msgId integer, referenceId integer, type text,voiceLength integer,content text,userId integer,userName text,localPath text)");
        sQLiteDatabase.execSQL("ALTER TABLE " + TABLE_MESSAGE_USER + " ADD COLUMN alias text");
    }

    @Override // android.database.sqlite.SQLiteOpenHelper
    public void onCreate(SQLiteDatabase sQLiteDatabase) {
        sQLiteDatabase.execSQL("CREATE TABLE " + TABLE_MESSAGE_LAST + " (id integer primary key autoincrement, sendUserId integer unique, msgLastId integer)");
        sQLiteDatabase.execSQL("CREATE TABLE " + TABLE_MESSAGE + " (id integer primary key autoincrement, msgId integer unique, userId integer, sendUserId integer,type text,voiceLength integer,content text,date text,revoke integer,reject integer,referenceId integer,androidRead integer,androidSend integer,localPath text,alias text)");
        sQLiteDatabase.execSQL("CREATE TABLE " + TABLE_MESSAGE_REFERENCE + "(id integer primary key autoincrement, msgId integer, referenceId integer, type text,voiceLength integer,content text,userId integer,userName text,localPath text)");
        sQLiteDatabase.execSQL("CREATE TABLE " + TABLE_MESSAGE_USER + " (id integer primary key autoincrement, userId integer unique, king integer, gradeIcon text, name text, gender integer,age text,constellation text,headIcon text,alias text)");
        sQLiteDatabase.execSQL("CREATE TABLE " + TABLE_TOPIC_DRAFT + " (id integer primary key autoincrement, userId integer , title text , content text , contentImage text , contentText text , date text ,  categorieId text)");
    }

    @Override // android.database.sqlite.SQLiteOpenHelper
    public void onUpgrade(SQLiteDatabase sQLiteDatabase, int i, int i2) {
        LogUtil.e("onUpgrade() oldVersion:" + i + ", newVersion:" + i2);
        while (true) {
            i++;
            if (i <= i2) {
                switch (i) {
                    case 7:
                        upgradeToVersion07(sQLiteDatabase);
                        break;
                    case 8:
                        upgradeToVersion08(sQLiteDatabase);
                        break;
                    case 9:
                        upgradeToVersion09(sQLiteDatabase);
                        break;
                    case 10:
                        upgradeToVersion10(sQLiteDatabase);
                        break;
                    case 11:
                        upgradeToVersion11(sQLiteDatabase);
                        break;
                    case 12:
                        upgradeToVersion12(sQLiteDatabase);
                        break;
                    case 13:
                        upgradeToVersion13(sQLiteDatabase);
                        break;
                }
            } else {
                return;
            }
        }
    }
}
