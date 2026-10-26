package com.hismith.hismithapp;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.KeyEvent;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.adapter.SearchFriendAdapter;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.UserSearchModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.IntentUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.SoftKeyboardUtil;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnLoadMoreListener;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class AddFriendActivity extends BaseActivity implements View.OnClickListener {
    private RetrofitRequest baseRequest;
    private SearchFriendAdapter mAdapter;
    private EditText mEditSearch;
    private ImageView mImgSearch;
    private LinearLayoutManager mLayoutManager;
    private LinearLayout mLayoutSearch;
    private RecyclerView mRecycler;
    private SmartRefreshLayout mRefreshLayout;
    private List<UserSearchModel.User> mSearchList;
    private ImageView mTitleBack;
    private TextView mTitleName;
    private TextView mTxtCancel;
    private String key = "";
    private int page = 1;
    private int totalPage = 1;

    static /* synthetic */ int access$608(AddFriendActivity addFriendActivity) {
        int i = addFriendActivity.page;
        addFriendActivity.page = i + 1;
        return i;
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        this.mRefreshLayout.setEnableRefresh(false);
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        this.mLayoutManager = linearLayoutManager;
        this.mRecycler.setLayoutManager(linearLayoutManager);
        this.mSearchList = new ArrayList();
        SearchFriendAdapter searchFriendAdapter = new SearchFriendAdapter(this, this.mSearchList);
        this.mAdapter = searchFriendAdapter;
        this.mRecycler.setAdapter(searchFriendAdapter);
    }

    private void initView() {
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleName = (TextView) findViewById(R.id.title_txt_name);
        this.mLayoutSearch = (LinearLayout) findViewById(R.id.add_layout_search);
        this.mImgSearch = (ImageView) findViewById(R.id.add_img_search);
        this.mEditSearch = (EditText) findViewById(R.id.add_edit_search);
        this.mTxtCancel = (TextView) findViewById(R.id.add_txt_cancel);
        this.mRefreshLayout = (SmartRefreshLayout) findViewById(R.id.add_refreshLayout);
        this.mRecycler = (RecyclerView) findViewById(R.id.add_recycler);
        LanguageUtil.setText(this, this.mTitleName, "searchUser");
        LanguageUtil.setHint(this, this.mEditSearch, "nicknameOrID");
        LanguageUtil.setText(this, this.mTxtCancel, "cancel");
    }

    private void setListener() {
        this.mTitleBack.setOnClickListener(this);
        this.mTxtCancel.setOnClickListener(this);
        this.mEditSearch.setOnFocusChangeListener(new View.OnFocusChangeListener() { // from class: com.hismith.hismithapp.AddFriendActivity.1
            @Override // android.view.View.OnFocusChangeListener
            public void onFocusChange(View view, boolean z) {
                AddFriendActivity addFriendActivity = AddFriendActivity.this;
                if (z) {
                    addFriendActivity.mImgSearch.setImageResource(R.mipmap.ic_community_search_focus);
                } else {
                    addFriendActivity.mImgSearch.setImageResource(R.mipmap.ic_community_search);
                }
            }
        });
        this.mEditSearch.setOnEditorActionListener(new TextView.OnEditorActionListener() { // from class: com.hismith.hismithapp.AddFriendActivity.2
            @Override // android.widget.TextView.OnEditorActionListener
            public boolean onEditorAction(TextView textView, int i, KeyEvent keyEvent) {
                if (i != 3) {
                    return false;
                }
                AddFriendActivity addFriendActivity = AddFriendActivity.this;
                addFriendActivity.key = addFriendActivity.mEditSearch.getText().toString().trim();
                AddFriendActivity.this.mSearchList.clear();
                AddFriendActivity.this.mAdapter.notifyDataSetChanged();
                AddFriendActivity.this.userSearch();
                LogUtil.e("搜索发送" + AddFriendActivity.this.key);
                SoftKeyboardUtil.hideSoftKeyboard(AddFriendActivity.this, textView);
                return true;
            }
        });
        this.mRefreshLayout.setOnLoadMoreListener(new OnLoadMoreListener() { // from class: com.hismith.hismithapp.AddFriendActivity.3
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                int i = AddFriendActivity.this.page;
                int i2 = AddFriendActivity.this.totalPage;
                AddFriendActivity addFriendActivity = AddFriendActivity.this;
                if (i >= i2) {
                    addFriendActivity.mRefreshLayout.finishLoadMore(1000);
                } else {
                    AddFriendActivity.access$608(addFriendActivity);
                    AddFriendActivity.this.userSearch();
                }
            }
        });
        this.mAdapter.setOnItemClickListener(new SearchFriendAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.AddFriendActivity.4
            @Override // com.hismith.hismithapp.adapter.SearchFriendAdapter.OnItemClickListener
            public void onItemClick(int i) {
                AddFriendActivity addFriendActivity = AddFriendActivity.this;
                IntentUtil.startUserInfoActivity(addFriendActivity, ((UserSearchModel.User) addFriendActivity.mSearchList.get(i)).getUserId());
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void userSearch() {
        if (TextUtils.isEmpty(this.key)) {
            return;
        }
        this.baseRequest.getApi().userSearch((String) SharedPreferencesUtil.get(this, Constans.Token), this.key, Integer.valueOf(this.page)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UserSearchModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.AddFriendActivity.5
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UserSearchModel userSearchModel) {
                if (userSearchModel.getSuccess() == 1) {
                    AddFriendActivity.this.totalPage = userSearchModel.getTotalPage();
                    if (userSearchModel.getUsers() != null) {
                        AddFriendActivity.this.mSearchList.addAll(Arrays.asList(userSearchModel.getUsers()));
                        AddFriendActivity.this.mAdapter.notifyDataSetChanged();
                    }
                }
            }
        });
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
        } else if (id == R.id.add_txt_cancel) {
            this.mEditSearch.setText("");
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_add_friend);
        initView();
        init();
        setListener();
    }
}
