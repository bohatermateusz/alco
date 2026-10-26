package com.hismith.hismithapp.fragment;

import android.content.ContentValues;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentTransaction;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.FriendShareAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.db.DatabaseHelper;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.MessageModel;
import com.hismith.hismithapp.model.MessageModel.Message;
import com.hismith.hismithapp.model.MsgSendModel;
import com.hismith.hismithapp.model.UserSearchModel;
import com.hismith.hismithapp.model.UserSearchModel.User;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.SoftKeyboardUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnLoadMoreListener;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/* loaded from: classes3.dex */
public class FriendShareFragment extends Fragment implements View.OnClickListener {
    private BaseActivity baseActivity;
    private RetrofitRequest baseRequest;
    private DatabaseHelper dbHepler;
    private SQLiteDatabase dbRead;
    private SQLiteDatabase dbWrite;
    private String key;
    private FriendShareAdapter mAdapter;
    private EditText mEditSearch;
    private List<UserSearchModel.User> mFriendList;
    private LinearLayoutManager mLayoutManager;
    private FrameLayout mLayoutSelect;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private List<UserSearchModel.User> mSearchList;
    private ImageView mTitleImgBack;
    private TextView mTitleTxtName;
    private TextView mTxtContent;
    private TextView mTxtContentLabel;
    private TextView mTxtRecent;
    private TextView mTxtSend;
    private int page = 1;
    private int totalPage = 1;
    private FriendShareAdapter.OnItemClickListener onItemClickListener = new FriendShareAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.fragment.FriendShareFragment.3
        @Override // com.hismith.hismithapp.adapter.FriendShareAdapter.OnItemClickListener
        public void onItemClick(int i) {
        }
    };

    static /* synthetic */ int access$1108(FriendShareFragment friendShareFragment) {
        int i = friendShareFragment.page;
        friendShareFragment.page = i + 1;
        return i;
    }

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

    private void getMessageUser() {
        Cursor query;
        try {
            String obj = SharedPreferencesUtil.get(this.baseActivity, Constans.deleteSession).toString();
            Cursor query2 = this.dbRead.query(DatabaseHelper.TABLE_MESSAGE, new String[]{"userId"}, "sendUserId = ?", new String[]{MyApplication.getApplication().getmUserModel().getUserId() + ""}, "userId", null, "msgId DESC");
            if (query2 == null || !query2.moveToFirst()) {
                return;
            }
            do {
                int i = query2.getInt(query2.getColumnIndex("userId"));
                if (!obj.contains("," + i + ",") && (query = this.dbRead.query(DatabaseHelper.TABLE_MESSAGE_USER, null, "userId = ?", new String[]{i + ""}, null, null, null)) != null && query.moveToFirst()) {
                    UserSearchModel userSearchModel = new UserSearchModel();
                    Objects.requireNonNull(userSearchModel);
                    UserSearchModel.User user = userSearchModel.new User();
                    user.setGender(query.getInt(query.getColumnIndex("gender")));
                    user.setHeadIcon(query.getString(query.getColumnIndex("headIcon")));
                    user.setKing(query.getInt(query.getColumnIndex("king")));
                    if (query.getColumnIndex("gradeIcon") != -1) {
                        user.setGradeIcon(query.getString(query.getColumnIndex("gradeIcon")));
                    }
                    user.setName(query.getString(query.getColumnIndex("name")));
                    user.setUserId(query.getInt(query.getColumnIndex("userId")));
                    this.mFriendList.add(user);
                    query.close();
                }
            } while (query2.moveToNext());
            this.mAdapter.notifyDataSetChanged();
            query2.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        DatabaseHelper databaseHelper = new DatabaseHelper(this.baseActivity);
        this.dbHepler = databaseHelper;
        this.dbWrite = databaseHelper.getWritableDatabase();
        this.dbRead = this.dbHepler.getReadableDatabase();
        this.mRefreshLayout.setEnableRefresh(false);
        this.mRefreshLayout.setEnableLoadMore(false);
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.baseActivity);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mFriendList = new ArrayList();
        this.mSearchList = new ArrayList();
        FriendShareAdapter friendShareAdapter = new FriendShareAdapter(this.baseActivity, this.mFriendList);
        this.mAdapter = friendShareAdapter;
        friendShareAdapter.setOnItemClickListener(this.onItemClickListener);
        this.mRecyclerView.setAdapter(this.mAdapter);
        getMessageUser();
    }

    private void initView(View view) {
        this.mTitleImgBack = (ImageView) view.findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) view.findViewById(R.id.title_txt_name);
        this.mTxtContentLabel = (TextView) view.findViewById(R.id.friend_share_txt_content_label);
        this.mTxtContent = (TextView) view.findViewById(R.id.friend_share_txt_content);
        this.mTxtRecent = (TextView) view.findViewById(R.id.friend_share_txt_recent);
        this.mTxtSend = (TextView) view.findViewById(R.id.friend_share_txt_send);
        this.mEditSearch = (EditText) view.findViewById(R.id.friend_share_edit_search);
        this.mRefreshLayout = (SmartRefreshLayout) view.findViewById(R.id.friend_share_refreshLayout);
        this.mRecyclerView = (RecyclerView) view.findViewById(R.id.friend_share_recycler_view);
        this.mLayoutSelect = (FrameLayout) view.findViewById(R.id.friend_share_layout_select);
        LanguageUtil.setText(this.baseActivity, this.mTitleTxtName, "remotePlay");
        LanguageUtil.setText(this.baseActivity, this.mTxtContentLabel, "shareContentTitle");
        LanguageUtil.setText(this.baseActivity, this.mTxtContent, "waitingRemoteConnection");
        LanguageUtil.setText(this.baseActivity, this.mTxtRecent, "recent");
        LanguageUtil.setHint(this.baseActivity, this.mEditSearch, "nicknameOrID");
        LanguageUtil.setText(this.baseActivity, this.mTxtSend, "send");
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void insertDBMessage(MessageModel.Message message) {
        ContentValues contentValues = new ContentValues();
        contentValues.put("msgId", Integer.valueOf(message.getMsgId()));
        contentValues.put("userId", Integer.valueOf(message.getUserId()));
        contentValues.put("sendUserId", Integer.valueOf(MyApplication.getApplication().getmUserModel().getUserId()));
        contentValues.put("type", message.getType());
        contentValues.put("voiceLength", Integer.valueOf(message.getVoiceLength()));
        contentValues.put(FirebaseAnalytics.Param.CONTENT, message.getContent());
        contentValues.put("date", message.getDate());
        contentValues.put("revoke", Integer.valueOf(message.getRevoke()));
        contentValues.put("androidSend", Integer.valueOf(message.getAndroidSend()));
        contentValues.put("androidRead", (Integer) 1);
        contentValues.put("referenceId", Integer.valueOf(message.getReferenceId()));
        this.dbWrite.replace(DatabaseHelper.TABLE_MESSAGE, null, contentValues);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void insertDBMessageUser(MessageModel.Message message) {
        ContentValues contentValues = new ContentValues();
        contentValues.put("userId", Integer.valueOf(message.getUserId()));
        contentValues.put("king", Integer.valueOf(message.getKing()));
        contentValues.put("gradeIcon", message.getGradeIcon());
        contentValues.put("name", message.getName());
        contentValues.put("gender", Integer.valueOf(message.getGender()));
        contentValues.put("age", message.getAge());
        contentValues.put("constellation", message.getConstellation());
        contentValues.put("headIcon", message.getHeadIcon());
        this.dbWrite.replace(DatabaseHelper.TABLE_MESSAGE_USER, null, contentValues);
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
        this.mTxtSend.setOnClickListener(this);
        this.mEditSearch.setOnEditorActionListener(new TextView.OnEditorActionListener() { // from class: com.hismith.hismithapp.fragment.FriendShareFragment.1
            @Override // android.widget.TextView.OnEditorActionListener
            public boolean onEditorAction(TextView textView, int i, KeyEvent keyEvent) {
                if (i != 3) {
                    return false;
                }
                FriendShareFragment.this.mAdapter.setSelectPosition(-1);
                FriendShareFragment.this.key = "";
                FriendShareFragment friendShareFragment = FriendShareFragment.this;
                friendShareFragment.key = friendShareFragment.mEditSearch.getText().toString().trim();
                boolean isEmpty = TextUtils.isEmpty(FriendShareFragment.this.key);
                FriendShareFragment friendShareFragment2 = FriendShareFragment.this;
                if (isEmpty) {
                    friendShareFragment2.mTxtRecent.setVisibility(0);
                    FriendShareFragment.this.mAdapter = new FriendShareAdapter(FriendShareFragment.this.baseActivity, FriendShareFragment.this.mFriendList);
                    FriendShareFragment.this.mAdapter.setOnItemClickListener(FriendShareFragment.this.onItemClickListener);
                    FriendShareFragment.this.mRecyclerView.setAdapter(FriendShareFragment.this.mAdapter);
                    FriendShareFragment.this.mRefreshLayout.setEnableLoadMore(false);
                } else {
                    friendShareFragment2.mTxtRecent.setVisibility(8);
                    FriendShareFragment.this.mSearchList.clear();
                    FriendShareFragment.this.mAdapter = new FriendShareAdapter(FriendShareFragment.this.baseActivity, FriendShareFragment.this.mSearchList);
                    FriendShareFragment.this.mAdapter.setOnItemClickListener(FriendShareFragment.this.onItemClickListener);
                    FriendShareFragment.this.mRecyclerView.setAdapter(FriendShareFragment.this.mAdapter);
                    FriendShareFragment friendShareFragment3 = FriendShareFragment.this;
                    friendShareFragment3.userSearch(friendShareFragment3.key);
                    FriendShareFragment.this.mRefreshLayout.setEnableLoadMore(true);
                }
                LogUtil.e("搜索发送:" + FriendShareFragment.this.key);
                SoftKeyboardUtil.hideSoftKeyboard(FriendShareFragment.this.baseActivity, textView);
                return true;
            }
        });
        this.mRefreshLayout.setOnLoadMoreListener(new OnLoadMoreListener() { // from class: com.hismith.hismithapp.fragment.FriendShareFragment.2
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                int i = FriendShareFragment.this.page;
                int i2 = FriendShareFragment.this.totalPage;
                FriendShareFragment friendShareFragment = FriendShareFragment.this;
                if (i >= i2) {
                    friendShareFragment.mRefreshLayout.finishLoadMore(1000);
                    return;
                }
                FriendShareFragment.access$1108(friendShareFragment);
                FriendShareFragment friendShareFragment2 = FriendShareFragment.this;
                friendShareFragment2.userSearch(friendShareFragment2.key);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void userSearch(String str) {
        if (TextUtils.isEmpty(str)) {
            this.mRefreshLayout.finishLoadMore();
            return;
        }
        Observable<UserSearchModel> observeOn = this.baseRequest.getApi().userSearch((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), str, Integer.valueOf(this.page)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<UserSearchModel>(baseActivity, baseActivity.progressDialog, true) { // from class: com.hismith.hismithapp.fragment.FriendShareFragment.5
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                FriendShareFragment.this.mRefreshLayout.finishLoadMore();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                FriendShareFragment.this.mRefreshLayout.finishLoadMore();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UserSearchModel userSearchModel) {
                if (userSearchModel.getSuccess() == 1) {
                    if (userSearchModel.getPage() == 1) {
                        FriendShareFragment.this.mSearchList.clear();
                    }
                    FriendShareFragment.this.totalPage = userSearchModel.getTotalPage();
                    if (userSearchModel.getUsers() != null) {
                        FriendShareFragment.this.mSearchList.addAll(Arrays.asList(userSearchModel.getUsers()));
                        FriendShareFragment.this.mAdapter.notifyDataSetChanged();
                    }
                }
            }
        });
    }

    public void msgSend(String str, final int i) {
        if (MyApplication.getApplication().getmUserModel() == null) {
            return;
        }
        Observable<MsgSendModel> observeOn = this.baseRequest.getApi().msgSend((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), String.valueOf(System.currentTimeMillis()), Integer.valueOf(this.mAdapter.getmFriendList().get(i).getUserId()), str, 0, "text").subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<MsgSendModel>(baseActivity, baseActivity.progressDialog, true) { // from class: com.hismith.hismithapp.fragment.FriendShareFragment.4
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(MsgSendModel msgSendModel) {
                if (msgSendModel.getSuccess() == 1) {
                    MessageModel messageModel = new MessageModel();
                    Objects.requireNonNull(messageModel);
                    MessageModel.Message message = messageModel.new Message();
                    message.setGender(FriendShareFragment.this.mAdapter.getmFriendList().get(i).getGender());
                    message.setHeadIcon(FriendShareFragment.this.mAdapter.getmFriendList().get(i).getHeadIcon());
                    message.setKing(FriendShareFragment.this.mAdapter.getmFriendList().get(i).getKing());
                    message.setGradeIcon(FriendShareFragment.this.mAdapter.getmFriendList().get(i).getGradeIcon());
                    message.setName(FriendShareFragment.this.mAdapter.getmFriendList().get(i).getName());
                    message.setAndroidSend(1);
                    message.setContent(msgSendModel.getContent());
                    message.setDate(msgSendModel.getDate());
                    message.setMsgId(msgSendModel.getMsgId());
                    message.setRevoke(0);
                    message.setType(msgSendModel.getType());
                    message.setUserId(msgSendModel.getUserId());
                    message.setVoiceLength(msgSendModel.getVoiceLength());
                    message.setStatus(0);
                    FriendShareFragment.this.insertDBMessage(message);
                    FriendShareFragment.this.insertDBMessageUser(message);
                    ToastUtil.showShortToastCenter(FriendShareFragment.this.baseActivity, LanguageUtil.getLanguageByKey(FriendShareFragment.this.baseActivity, "sended"));
                    FragmentTransaction beginTransaction = FriendShareFragment.this.getFragmentManager().beginTransaction();
                    beginTransaction.hide(FriendShareFragment.this);
                    beginTransaction.commit();
                }
            }
        });
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            this.baseActivity.finish();
            return;
        }
        if (id == R.id.friend_share_txt_send) {
            String languageByKey = LanguageUtil.getLanguageByKey(this.baseActivity, "waitingRemoteConnection");
            String obj = this.mTxtContent.getText().toString();
            if (languageByKey.equals(obj)) {
                ToastUtil.showShortToastCenter(this.baseActivity, languageByKey);
            } else if (this.mAdapter.getSelectPosition() != -1) {
                msgSend(obj, this.mAdapter.getSelectPosition());
            }
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.activity_friend_share, (ViewGroup) null);
        initView(inflate);
        init();
        setListener();
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        closeDB();
    }
}
