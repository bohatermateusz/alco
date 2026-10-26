package com.hismith.hismithapp.fragment;

import android.content.ContentValues;
import android.content.Intent;
import android.database.sqlite.SQLiteDatabase;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.recyclerview.widget.DividerItemDecoration;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.ChatActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.ChatUpAdapter;
import com.hismith.hismithapp.db.DatabaseHelper;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.ChatUpModel;
import com.hismith.hismithapp.model.FirstChatModel;
import com.hismith.hismithapp.model.MessageModel;
import com.hismith.hismithapp.model.MessageModel.Message;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshLoadMoreListener;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/* loaded from: classes3.dex */
public class PersonalChatUpFragment extends LazyLoadFragment {
    public static final int WHAT_NOTIFYDATA = 1;
    private BaseActivity baseActivity;
    private RetrofitRequest baseRequest;
    private DatabaseHelper dbHepler;
    private SQLiteDatabase dbWrite;
    private ChatUpAdapter mAdapter;
    private LinearLayoutManager mLinearLayoutManager;
    private List<FirstChatModel> mList;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private int page = 1;
    private int totalPage = 1;
    public Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.PersonalChatUpFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what == 1 && PersonalChatUpFragment.this.mRecyclerView != null) {
                PersonalChatUpFragment.this.mRecyclerView.scrollBy(0, 1);
                PersonalChatUpFragment.this.mRecyclerView.scrollBy(0, -1);
            }
            return false;
        }
    });

    static /* synthetic */ int access$108(PersonalChatUpFragment personalChatUpFragment) {
        int i = personalChatUpFragment.page;
        personalChatUpFragment.page = i + 1;
        return i;
    }

    private void closeDB() {
        SQLiteDatabase sQLiteDatabase = this.dbWrite;
        if (sQLiteDatabase != null) {
            sQLiteDatabase.close();
        }
        DatabaseHelper databaseHelper = this.dbHepler;
        if (databaseHelper != null) {
            databaseHelper.close();
        }
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        DatabaseHelper databaseHelper = new DatabaseHelper(this.baseActivity);
        this.dbHepler = databaseHelper;
        this.dbWrite = databaseHelper.getWritableDatabase();
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.baseActivity);
        this.mLinearLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mRecyclerView.addItemDecoration(new DividerItemDecoration(this.baseActivity, 1));
        this.mList = new ArrayList();
        ChatUpAdapter chatUpAdapter = new ChatUpAdapter(this.baseActivity, this.mList);
        this.mAdapter = chatUpAdapter;
        this.mRecyclerView.setAdapter(chatUpAdapter);
        this.mRefreshLayout.autoRefresh();
    }

    private void initView(View view) {
        this.mRefreshLayout = (SmartRefreshLayout) view.findViewById(R.id.personal_chatup_refreshLayout);
        this.mRecyclerView = (RecyclerView) view.findViewById(R.id.personal_chatup_recycler_view);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void insertDBMessageUser(ChatUpModel chatUpModel) {
        for (int i = 0; i < chatUpModel.getFirstChat().length; i++) {
            try {
                FirstChatModel firstChatModel = chatUpModel.getFirstChat()[i];
                ContentValues contentValues = new ContentValues();
                contentValues.put("userId", Integer.valueOf(firstChatModel.getUserId()));
                contentValues.put("king", Integer.valueOf(firstChatModel.getKing()));
                contentValues.put("gradeIcon", firstChatModel.getGradeIcon());
                contentValues.put("name", firstChatModel.getName());
                contentValues.put("gender", Integer.valueOf(firstChatModel.getGender()));
                contentValues.put("age", firstChatModel.getAge());
                contentValues.put("constellation", firstChatModel.getConstellation());
                contentValues.put("headIcon", firstChatModel.getHeadIcon());
                this.dbWrite.replace(DatabaseHelper.TABLE_MESSAGE_USER, null, contentValues);
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void msgFirstChat(int i) {
        this.baseRequest.getApi().msgFirstChat((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ChatUpModel>(this.baseActivity, null, false) { // from class: com.hismith.hismithapp.fragment.PersonalChatUpFragment.4
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                PersonalChatUpFragment.this.mRefreshLayout.finishRefresh();
                PersonalChatUpFragment.this.mRefreshLayout.finishLoadMore();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                PersonalChatUpFragment.this.mRefreshLayout.finishRefresh();
                PersonalChatUpFragment.this.mRefreshLayout.finishLoadMore();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ChatUpModel chatUpModel) {
                try {
                    if (chatUpModel.getSuccess() == 1) {
                        if (chatUpModel.getPage() == 1) {
                            PersonalChatUpFragment.this.mList.clear();
                        }
                        PersonalChatUpFragment.this.totalPage = chatUpModel.getTotalPage();
                        if (chatUpModel.getFirstChat() != null) {
                            PersonalChatUpFragment.this.mList.addAll(Arrays.asList(chatUpModel.getFirstChat()));
                            PersonalChatUpFragment.this.insertDBMessageUser(chatUpModel);
                        }
                        if (PersonalChatUpFragment.this.mAdapter != null) {
                            PersonalChatUpFragment.this.mAdapter.setShowEmptyView(true);
                            PersonalChatUpFragment.this.mAdapter.notifyDataSetChanged();
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    private void setListener() {
        this.mRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.fragment.PersonalChatUpFragment.2
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                int i = PersonalChatUpFragment.this.page;
                int i2 = PersonalChatUpFragment.this.totalPage;
                PersonalChatUpFragment personalChatUpFragment = PersonalChatUpFragment.this;
                if (i >= i2) {
                    personalChatUpFragment.mRefreshLayout.finishLoadMore(1000);
                    return;
                }
                PersonalChatUpFragment.access$108(personalChatUpFragment);
                PersonalChatUpFragment personalChatUpFragment2 = PersonalChatUpFragment.this;
                personalChatUpFragment2.msgFirstChat(personalChatUpFragment2.page);
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                PersonalChatUpFragment.this.page = 1;
                PersonalChatUpFragment personalChatUpFragment = PersonalChatUpFragment.this;
                personalChatUpFragment.msgFirstChat(personalChatUpFragment.page);
            }
        });
        this.mAdapter.setOnItemCLickListener(new ChatUpAdapter.OnItemCLickListener() { // from class: com.hismith.hismithapp.fragment.PersonalChatUpFragment.3
            @Override // com.hismith.hismithapp.adapter.ChatUpAdapter.OnItemCLickListener
            public void onItemClick(int i) {
                UserModel userModel = new UserModel();
                userModel.setConstellation(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getConstellation());
                userModel.setGender(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getGender());
                userModel.setHeadIcon(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getHeadIcon());
                userModel.setKing(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getKing());
                userModel.setGradeIcon(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getGradeIcon());
                userModel.setName(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getName());
                userModel.setUserId(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getUserId());
                MessageModel messageModel = new MessageModel();
                Objects.requireNonNull(messageModel);
                MessageModel.Message message = messageModel.new Message();
                message.setContent(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getContent());
                message.setDate(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getDate());
                message.setType(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getType());
                message.setUserId(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getUserId());
                message.setVoiceLength(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getVoiceLength());
                message.setConstellation(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getConstellation());
                message.setGender(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getGender());
                message.setHeadIcon(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getHeadIcon());
                message.setKing(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getKing());
                message.setGradeIcon(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getGradeIcon());
                message.setName(((FirstChatModel) PersonalChatUpFragment.this.mList.get(i)).getName());
                message.setStatus(0);
                Intent intent = new Intent(PersonalChatUpFragment.this.baseActivity, (Class<?>) ChatActivity.class);
                intent.putExtra(Constans.UserModel, userModel);
                intent.putExtra("messageModel", message);
                intent.putExtra("firstChat", 1);
                intent.putExtra("refreshUser", true);
                PersonalChatUpFragment.this.startActivity(intent);
            }
        });
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    public void loadData() {
        SmartRefreshLayout smartRefreshLayout = this.mRefreshLayout;
        if (smartRefreshLayout != null) {
            smartRefreshLayout.autoRefresh();
        }
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_personal_chatup, (ViewGroup) null);
        initView(inflate);
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        closeDB();
    }
}
