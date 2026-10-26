package com.hismith.hismithapp.fragment;

import android.content.BroadcastReceiver;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.ChatActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.ChatUpAdapter;
import com.hismith.hismithapp.adapter.MessageAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.db.DatabaseHelper;
import com.hismith.hismithapp.model.MessageModel;
import com.hismith.hismithapp.model.MessageModel.Message;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshLoadMoreListener;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/* loaded from: classes3.dex */
public class PersonalMessageFragment extends LazyLoadFragment {
    public static final int REQUEST_MESSAGE_NEW = 114;
    private BaseActivity baseActivity;
    private DatabaseHelper dbHepler;
    private SQLiteDatabase dbRead;
    private SQLiteDatabase dbWrite;
    private MessageAdapter mAdapter;
    private LinearLayoutManager mLinearLayoutManager;
    private List<MessageModel.Message> mList;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private int pageCount = 0;
    private int page = 1;
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.PersonalMessageFragment.4
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            LogUtil.e("PersonalFragment Receiver action:" + action);
            if (action.equals(Constans.RECEIVER_MESSAGE_NEW)) {
                String stringExtra = intent.getStringExtra("revokeId");
                intent.getStringExtra("rejectId");
                int intExtra = intent.getIntExtra("newMessage", 0);
                intent.getIntExtra("newMessageTotal", 0);
                LogUtil.e("mReceiver newMessage:" + intExtra + ", newFirstChat:" + intent.getIntExtra("newFirstChat", 0) + ", newInteractive:" + intent.getIntExtra("newInteractive", 0) + ", newNotice:" + intent.getIntExtra("newNotice", 0));
                if (intExtra > 0 || !TextUtils.isEmpty(stringExtra)) {
                    PersonalMessageFragment.this.refreshMessage();
                }
            }
        }
    };

    private void closeDB() {
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

    /* JADX INFO: Access modifiers changed from: private */
    public void getDBMessage() {
        try {
            String obj = SharedPreferencesUtil.get(this.baseActivity, Constans.deleteSession).toString();
            Cursor rawQuery = this.dbRead.rawQuery("SELECT * FROM " + DatabaseHelper.TABLE_MESSAGE + " AS a INNER JOIN (SELECT MAX(msgId) AS msgId, SUM(androidRead==0) as unreadCount FROM " + DatabaseHelper.TABLE_MESSAGE + " WHERE revoke = 0 and sendUserId = ?  GROUP BY userId) AS b ON a.msgId=b.msgId ORDER BY a.msgId DESC;", new String[]{MyApplication.getApplication().getmUserModel().getUserId() + ""});
            while (rawQuery != null && rawQuery.moveToNext()) {
                MessageModel messageModel = new MessageModel();
                Objects.requireNonNull(messageModel);
                MessageModel.Message message = messageModel.new Message();
                int i = rawQuery.getInt(rawQuery.getColumnIndexOrThrow("userId"));
                if (!obj.contains("," + i + ",")) {
                    message.setMsgId(rawQuery.getInt(rawQuery.getColumnIndexOrThrow("msgId")));
                    message.setUserId(i);
                    message.setVoiceLength(rawQuery.getInt(rawQuery.getColumnIndexOrThrow("voiceLength")));
                    message.setType(rawQuery.getString(rawQuery.getColumnIndexOrThrow("type")));
                    message.setRevoke(rawQuery.getInt(rawQuery.getColumnIndexOrThrow("revoke")));
                    message.setReject(rawQuery.getInt(rawQuery.getColumnIndexOrThrow("reject")));
                    message.setDate(rawQuery.getString(rawQuery.getColumnIndexOrThrow("date")));
                    message.setContent(rawQuery.getString(rawQuery.getColumnIndexOrThrow(FirebaseAnalytics.Param.CONTENT)));
                    message.setAndroidRead(rawQuery.getInt(rawQuery.getColumnIndexOrThrow("androidRead")));
                    message.setAndroidSend(rawQuery.getInt(rawQuery.getColumnIndexOrThrow("androidSend")));
                    message.setAndroidReadCount(rawQuery.getInt(rawQuery.getColumnIndexOrThrow("unreadCount")));
                    Cursor query = this.dbRead.query(DatabaseHelper.TABLE_MESSAGE_USER, null, "userId = ?", new String[]{i + ""}, null, null, null);
                    if (query != null && query.moveToFirst()) {
                        message.setName(query.getString(query.getColumnIndexOrThrow("name")));
                        message.setKing(query.getInt(query.getColumnIndexOrThrow("king")));
                        if (query.getColumnIndex("gradeIcon") != -1) {
                            message.setGradeIcon(query.getString(query.getColumnIndexOrThrow("gradeIcon")));
                        }
                        message.setHeadIcon(query.getString(query.getColumnIndexOrThrow("headIcon")));
                        message.setGender(query.getInt(query.getColumnIndexOrThrow("gender")));
                        message.setConstellation(query.getString(query.getColumnIndexOrThrow("constellation")));
                        message.setAlias(query.getString(query.getColumnIndexOrThrow("alias")));
                        LogUtil.e("返回列表：" + message.getName() + ", alias:" + message.getAlias());
                        query.close();
                    }
                    this.mList.add(message);
                }
            }
            MessageAdapter messageAdapter = this.mAdapter;
            if (messageAdapter != null) {
                messageAdapter.setShowEmptyView(true);
                this.mAdapter.notifyDataSetChanged();
            }
            if (rawQuery != null) {
                rawQuery.close();
            }
            Intent intent = new Intent(Constans.RECEIVER_MESSAGE_READ);
            intent.putExtra("isUnRead", hasUnRead());
            this.baseActivity.sendBroadcast(intent);
        } catch (Exception e) {
            e.printStackTrace();
        }
        this.mRefreshLayout.finishRefresh();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public boolean hasUnRead() {
        for (int i = 0; i < this.mList.size(); i++) {
            if (this.mList.get(i).getAndroidRead() == 0) {
                return true;
            }
        }
        return false;
    }

    private void init() {
        DatabaseHelper databaseHelper = new DatabaseHelper(this.baseActivity);
        this.dbHepler = databaseHelper;
        this.dbRead = databaseHelper.getReadableDatabase();
        this.dbWrite = this.dbHepler.getWritableDatabase();
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.baseActivity);
        this.mLinearLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mList = new ArrayList();
        MessageAdapter messageAdapter = new MessageAdapter(this.baseActivity, this.mList);
        this.mAdapter = messageAdapter;
        this.mRecyclerView.setAdapter(messageAdapter);
        this.mRefreshLayout.setEnableLoadMore(false);
        this.mRefreshLayout.autoRefresh();
    }

    private void initView(View view) {
        this.mRefreshLayout = (SmartRefreshLayout) view.findViewById(R.id.personal_message_refreshLayout);
        this.mRecyclerView = (RecyclerView) view.findViewById(R.id.personal_message_recycler_view);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshMessage() {
        if (!this.isLoad || MyApplication.getApplication().getmUserModel() == null) {
            return;
        }
        List<MessageModel.Message> list = this.mList;
        if (list != null) {
            list.clear();
        }
        getDBMessage();
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_MESSAGE_NEW);
        AppUtil.registerReceiver(this.baseActivity, this.mReceiver, intentFilter);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void removeSession(int i) {
        try {
            String obj = SharedPreferencesUtil.get(this.baseActivity, Constans.deleteSession).toString();
            if (TextUtils.isEmpty(obj)) {
                SharedPreferencesUtil.put(this.baseActivity, Constans.deleteSession, "," + this.mList.get(i).getUserId() + ",");
            } else if (!obj.contains("," + this.mList.get(i).getUserId() + ",")) {
                SharedPreferencesUtil.put(this.baseActivity, Constans.deleteSession, obj + "," + this.mList.get(i).getUserId() + ",");
            }
            updateDBManagerRead(this.mList.get(i).getUserId());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void setListener() {
        this.mRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.fragment.PersonalMessageFragment.1
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                UserModel userModel = MyApplication.getApplication().getmUserModel();
                PersonalMessageFragment personalMessageFragment = PersonalMessageFragment.this;
                if (userModel == null) {
                    personalMessageFragment.mRefreshLayout.finishRefresh(1000);
                } else {
                    personalMessageFragment.mList.clear();
                    PersonalMessageFragment.this.getDBMessage();
                }
            }
        });
        this.mAdapter.setOnItemCLickListener(new ChatUpAdapter.OnItemCLickListener() { // from class: com.hismith.hismithapp.fragment.PersonalMessageFragment.2
            @Override // com.hismith.hismithapp.adapter.ChatUpAdapter.OnItemCLickListener
            public void onItemClick(int i) {
                UserModel userModel = new UserModel();
                userModel.setConstellation(((MessageModel.Message) PersonalMessageFragment.this.mList.get(i)).getConstellation());
                userModel.setGender(((MessageModel.Message) PersonalMessageFragment.this.mList.get(i)).getGender());
                userModel.setHeadIcon(((MessageModel.Message) PersonalMessageFragment.this.mList.get(i)).getHeadIcon());
                userModel.setKing(((MessageModel.Message) PersonalMessageFragment.this.mList.get(i)).getKing());
                userModel.setGradeIcon(((MessageModel.Message) PersonalMessageFragment.this.mList.get(i)).getGradeIcon());
                userModel.setName(((MessageModel.Message) PersonalMessageFragment.this.mList.get(i)).getName());
                userModel.setAlias(((MessageModel.Message) PersonalMessageFragment.this.mList.get(i)).getAlias());
                userModel.setUserId(((MessageModel.Message) PersonalMessageFragment.this.mList.get(i)).getUserId());
                Intent intent = new Intent(PersonalMessageFragment.this.baseActivity, (Class<?>) ChatActivity.class);
                intent.putExtra(Constans.UserModel, userModel);
                intent.putExtra("firstChat", 0);
                intent.putExtra("refreshUser", true);
                PersonalMessageFragment.this.startActivityForResult(intent, PersonalMessageFragment.REQUEST_MESSAGE_NEW);
                Intent intent2 = new Intent(Constans.RECEIVER_MESSAGE_READ);
                intent2.putExtra("isUnRead", PersonalMessageFragment.this.hasUnRead());
                PersonalMessageFragment.this.baseActivity.sendBroadcast(intent2);
            }
        });
        this.mAdapter.setOnDeleteClickListener(new MessageAdapter.OnDeleteClickListener() { // from class: com.hismith.hismithapp.fragment.PersonalMessageFragment.3
            @Override // com.hismith.hismithapp.adapter.MessageAdapter.OnDeleteClickListener
            public void onDeleteClick(int i) {
                PersonalMessageFragment.this.removeSession(i);
                PersonalMessageFragment.this.mList.remove(i);
                PersonalMessageFragment.this.mAdapter.notifyItemRemoved(i);
                PersonalMessageFragment.this.mAdapter.notifyItemRangeChanged(i, PersonalMessageFragment.this.mList.size());
                Intent intent = new Intent(Constans.RECEIVER_MESSAGE_READ);
                intent.putExtra("isUnRead", PersonalMessageFragment.this.hasUnRead());
                PersonalMessageFragment.this.baseActivity.sendBroadcast(intent);
            }
        });
    }

    private void unRegisterReceiver() {
        try {
            this.baseActivity.unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void updateDBManagerRead(int i) {
        try {
            ContentValues contentValues = new ContentValues();
            contentValues.put("androidRead", (Integer) 1);
            this.dbWrite.update(DatabaseHelper.TABLE_MESSAGE, contentValues, "userId = ? and sendUserId = ?", new String[]{i + "", MyApplication.getApplication().getmUserModel().getUserId() + ""});
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // androidx.fragment.app.Fragment
    public void onActivityResult(int i, int i2, Intent intent) {
        super.onActivityResult(i, i2, intent);
        if (i == 114 && this.isLoad && MyApplication.getApplication().getmUserModel() != null) {
            this.mList.clear();
            getDBMessage();
        }
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_personal_message, (ViewGroup) null);
        initView(inflate);
        registerReceiver();
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        closeDB();
        unRegisterReceiver();
    }

    @Override // androidx.fragment.app.Fragment
    public void onResume() {
        super.onResume();
    }
}
