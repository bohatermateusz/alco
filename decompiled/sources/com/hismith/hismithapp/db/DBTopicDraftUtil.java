package com.hismith.hismithapp.db;

import android.content.ContentValues;
import android.database.Cursor;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.model.TopicDraftModel;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.StringUtils;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class DBTopicDraftUtil extends BaseDBUtil {
    public static void deleteDBTopicDraftById(int i) {
        if (i > 0) {
            deleteData(DatabaseHelper.TABLE_TOPIC_DRAFT, " id = ? ", new String[]{i + ""});
        }
    }

    public static List<TopicDraftModel> getDraftByUserId(int i) {
        LogUtil.e("getDraftByUserId(" + i + ")");
        open();
        ArrayList arrayList = new ArrayList();
        Cursor query = dbRead.query(DatabaseHelper.TABLE_TOPIC_DRAFT, null, "userId = ?", new String[]{i + ""}, null, null, "date DESC");
        if (query != null && query.moveToFirst()) {
            do {
                TopicDraftModel topicDraftModel = new TopicDraftModel();
                topicDraftModel.setId(query.getInt(query.getColumnIndexOrThrow("id")));
                topicDraftModel.setUserId(query.getInt(query.getColumnIndexOrThrow("userId")));
                topicDraftModel.setTitle(query.getString(query.getColumnIndexOrThrow("title")));
                topicDraftModel.setContent(query.getString(query.getColumnIndexOrThrow(FirebaseAnalytics.Param.CONTENT)));
                topicDraftModel.setContentPhoto(query.getString(query.getColumnIndexOrThrow("contentImage")).split(","));
                topicDraftModel.setContentText(query.getString(query.getColumnIndexOrThrow("contentText")));
                topicDraftModel.setDate(query.getString(query.getColumnIndexOrThrow("date")));
                topicDraftModel.setCategorieId(query.getString(query.getColumnIndexOrThrow("categorieId")));
                arrayList.add(topicDraftModel);
            } while (query.moveToNext());
        }
        if (query != null) {
            query.close();
        }
        closeAll();
        return arrayList;
    }

    public static void insertDBTopicDraft(TopicDraftModel topicDraftModel) {
        LogUtil.e("insertDBTopicDraft()");
        ContentValues contentValues = new ContentValues();
        if (topicDraftModel.getId() != 0) {
            contentValues.put("id", Integer.valueOf(topicDraftModel.getId()));
        }
        contentValues.put("userId", Integer.valueOf(MyApplication.getApplication().getmUserModel().getUserId()));
        contentValues.put("title", topicDraftModel.getTitle());
        contentValues.put(FirebaseAnalytics.Param.CONTENT, topicDraftModel.getContent());
        contentValues.put("contentImage", StringUtils.join(topicDraftModel.getContentPhoto(), ","));
        contentValues.put("contentText", topicDraftModel.getContentText());
        contentValues.put("date", topicDraftModel.getDate());
        contentValues.put("categorieId", topicDraftModel.getCategorieId());
        insertData(DatabaseHelper.TABLE_TOPIC_DRAFT, contentValues);
    }
}
