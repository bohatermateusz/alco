package com.hismith.hismithapp.db;

import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.text.TextUtils;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.model.MessageModel;
import com.hismith.hismithapp.model.ReferenceMsg;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;

/* loaded from: classes3.dex */
public class DBMessageUtil {
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

    public static int getMessageCountByUser(int i) {
        Exception exc;
        int i2 = 0;
        try {
            open();
            Cursor query = dbRead.query(DatabaseHelper.TABLE_MESSAGE, new String[]{"count(msgId)"}, "userId = ? and sendUserId = ?", new String[]{i + "", MyApplication.getApplication().getmUserModel().getUserId() + ""}, null, null, null);
            if (query != null && query.moveToFirst()) {
                int i3 = 0;
                do {
                    try {
                        i3 = query.getInt(0);
                    } catch (Exception e) {
                        exc = e;
                        i2 = i3;
                        exc.printStackTrace();
                        return i2;
                    }
                } while (query.moveToNext());
                i2 = i3;
            }
            if (query != null) {
                query.close();
            }
            closeAll();
            return i2;
        } catch (Exception e2) {
            exc = e2;
        }
    }

    public static int getMessageLastId() {
        int i = 0;
        try {
            open();
            Cursor query = dbRead.query(DatabaseHelper.TABLE_MESSAGE, new String[]{"max(msgId)"}, "sendUserId = ?", new String[]{MyApplication.getApplication().getmUserModel().getUserId() + ""}, null, null, null);
            if (query != null && query.moveToFirst()) {
                int i2 = 0;
                do {
                    try {
                        i2 = query.getInt(0);
                    } catch (Exception e) {
                        e = e;
                        i = i2;
                        e.printStackTrace();
                        return i;
                    }
                } while (query.moveToNext());
                i = i2;
            }
            if (query != null) {
                query.close();
            }
            closeAll();
            return i;
        } catch (Exception e2) {
            e = e2;
        }
    }

    public static int getMessageUnReadCount() {
        open();
        int i = 0;
        Cursor query = dbRead.query(DatabaseHelper.TABLE_MESSAGE, new String[]{"count(msgId)"}, "revoke = 0 and androidRead = ? and sendUserId = ?", new String[]{"0", MyApplication.getApplication().getmUserModel().getUserId() + ""}, null, null, null);
        if (query != null && query.moveToFirst()) {
            do {
                i = query.getInt(query.getColumnIndexOrThrow("count(msgId)"));
            } while (query.moveToNext());
        }
        if (query != null) {
            query.close();
        }
        closeAll();
        return i;
    }

    public static int getSendByMessageId(int i) {
        int i2 = -1;
        try {
            open();
            Cursor query = dbRead.query(DatabaseHelper.TABLE_MESSAGE, new String[]{"androidSend"}, "msgId = ?", new String[]{i + ""}, null, null, null);
            if (query != null && query.moveToFirst()) {
                do {
                    i2 = query.getInt(0);
                } while (query.moveToNext());
            }
            if (query != null) {
                query.close();
            }
            closeAll();
            return i2;
        } catch (Exception e) {
            e.printStackTrace();
            return i2;
        }
    }

    public static void insertDBMessage(Context context, MessageModel.Message message) {
        try {
            String obj = SharedPreferencesUtil.get(context, Constans.deleteSession).toString();
            if (!TextUtils.isEmpty(obj) && obj.contains("," + message.getUserId() + ",")) {
                SharedPreferencesUtil.put(context, Constans.deleteSession, obj.replace("," + message.getUserId() + ",", ""));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        open();
        ContentValues contentValues = new ContentValues();
        contentValues.put("msgId", Integer.valueOf(message.getMsgId()));
        contentValues.put("userId", Integer.valueOf(message.getUserId()));
        contentValues.put("sendUserId", Integer.valueOf(MyApplication.getApplication().getmUserModel().getUserId()));
        contentValues.put("type", message.getType());
        contentValues.put("voiceLength", Integer.valueOf(message.getVoiceLength()));
        contentValues.put(FirebaseAnalytics.Param.CONTENT, message.getContent());
        contentValues.put("localPath", message.getLocalPath());
        contentValues.put("date", message.getDate());
        contentValues.put("revoke", Integer.valueOf(message.getRevoke()));
        contentValues.put("androidSend", Integer.valueOf(message.getAndroidSend()));
        contentValues.put("androidRead", (Integer) 1);
        contentValues.put("referenceId", Integer.valueOf(message.getReferenceId()));
        dbWrite.replace(DatabaseHelper.TABLE_MESSAGE, null, contentValues);
        closeAll();
    }

    public static void insertDBMessageLast(MessageModel.Message message) {
        open();
        ContentValues contentValues = new ContentValues();
        contentValues.put("sendUserId", Integer.valueOf(MyApplication.getApplication().getmUserModel().getUserId()));
        contentValues.put("msgLastId", Integer.valueOf(message.getMsgId()));
        dbWrite.replace(DatabaseHelper.TABLE_MESSAGE_LAST, null, contentValues);
        closeAll();
    }

    public static void insertMessageReference(int i, ReferenceMsg referenceMsg) {
        open();
        ContentValues contentValues = new ContentValues();
        contentValues.put("msgId", Integer.valueOf(i));
        contentValues.put("referenceId", Integer.valueOf(referenceMsg.getReferenceId()));
        contentValues.put("type", referenceMsg.getType());
        contentValues.put("voiceLength", Integer.valueOf(referenceMsg.getVoiceLength()));
        contentValues.put(FirebaseAnalytics.Param.CONTENT, referenceMsg.getContent());
        contentValues.put("localPath", referenceMsg.getLocalPath());
        contentValues.put("userId", Integer.valueOf(referenceMsg.getUserId()));
        contentValues.put("userName", referenceMsg.getUserName());
        dbWrite.replace(DatabaseHelper.TABLE_MESSAGE_REFERENCE, null, contentValues);
        closeAll();
    }

    private static void open() {
        DatabaseHelper databaseHelper = new DatabaseHelper(MyApplication.getApplication());
        dbHepler = databaseHelper;
        dbWrite = databaseHelper.getWritableDatabase();
        dbRead = dbHepler.getReadableDatabase();
    }

    public static void updateDBAlias(int i, int i2, String str) {
        open();
        ContentValues contentValues = new ContentValues();
        contentValues.put("alias", str);
        dbWrite.update(DatabaseHelper.TABLE_MESSAGE, contentValues, "msgId = (select max(msgId) from " + DatabaseHelper.TABLE_MESSAGE + " where userId = ? and sendUserId = ?)", new String[]{i + "", i2 + ""});
        closeAll();
    }

    public static void updateDBReject(Context context, String str, String str2) {
        open();
        ContentValues contentValues = new ContentValues();
        contentValues.put("reject", str2);
        dbWrite.update(DatabaseHelper.TABLE_MESSAGE, contentValues, "msgId = ?", new String[]{str});
        closeAll();
        Intent intent = new Intent(Constans.RECEIVER_MESSAGE_NEW);
        intent.putExtra("msgId", str);
        context.sendBroadcast(intent);
    }

    public static void updateDBRemoteTime(Context context, String str, int i) {
        try {
            open();
            ContentValues contentValues = new ContentValues();
            contentValues.put("voiceLength", Integer.valueOf(i));
            dbWrite.update(DatabaseHelper.TABLE_MESSAGE, contentValues, "msgId = ?", new String[]{str});
            closeAll();
            Intent intent = new Intent(Constans.RECEIVER_MESSAGE_NEW);
            intent.putExtra("msgId", str);
            context.sendBroadcast(intent);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
