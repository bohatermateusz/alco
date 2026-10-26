package com.hismith.hismithapp;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.DividerItemDecoration;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.listener.CustomClickListener;
import com.hismith.hismithapp.model.LanguageModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.List;

/* loaded from: classes3.dex */
public class LanguageActivity extends BaseActivity implements View.OnClickListener {
    private RetrofitRequest baseRequest;
    private LinearLayoutManager mLayoutManager;
    private RecyclerView mRecyclerView;
    private ImageView mTitleBack;
    private TextView mTitleTxtName;

    class LanguageAdapter extends RecyclerView.Adapter<MyViewHolder> {
        private Context mContext;
        private List<LanguageModel> mList;

        class MyViewHolder extends RecyclerView.ViewHolder {
            private ImageView mItemImgLanguageConfirm;
            private TextView mItemTxtLanguage;

            public MyViewHolder(View view) {
                super(view);
                this.mItemTxtLanguage = (TextView) view.findViewById(R.id.item_txt_language);
                this.mItemImgLanguageConfirm = (ImageView) view.findViewById(R.id.item_img_language_confirm);
            }
        }

        public LanguageAdapter(Context context, List<LanguageModel> list) {
            this.mContext = context;
            this.mList = list;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void changeLanguage(String str) {
            SharedPreferencesUtil.put(this.mContext, Constans.Language, str);
            notifyDataSetChanged();
            MyApplication.getApplication().initLanguage();
            LanguageActivity.this.appManager.finishAllActivity();
            LanguageActivity.this.startActivity(new Intent(LanguageActivity.this, (Class<?>) MainActivity.class));
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public int getItemCount() {
            return this.mList.size();
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public void onBindViewHolder(MyViewHolder myViewHolder, final int i) {
            myViewHolder.mItemImgLanguageConfirm.setVisibility(8);
            if (this.mList.get(i).getIso().equals(LanguageUtil.getCurrLanguage(this.mContext))) {
                myViewHolder.mItemImgLanguageConfirm.setVisibility(0);
            }
            myViewHolder.mItemTxtLanguage.setText(this.mList.get(i).getName());
            myViewHolder.mItemTxtLanguage.setOnClickListener(new CustomClickListener() { // from class: com.hismith.hismithapp.LanguageActivity.LanguageAdapter.1
                @Override // com.hismith.hismithapp.listener.CustomClickListener
                public void onSingleClick(View view) {
                    if (((LanguageModel) LanguageAdapter.this.mList.get(i)).getIso().equals(LanguageUtil.getCurrLanguage(LanguageAdapter.this.mContext))) {
                        return;
                    }
                    LanguageActivity.this.baseRequest.getApi().setUserLang((String) SharedPreferencesUtil.get(LanguageAdapter.this.mContext, Constans.Token), ((LanguageModel) LanguageAdapter.this.mList.get(i)).getIso()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(LanguageActivity.this, LanguageActivity.this.progressDialog) { // from class: com.hismith.hismithapp.LanguageActivity.LanguageAdapter.1.1
                        @Override // com.hismith.hismithapp.http.RxSubscribe
                        protected void onSuccess(HttpModel httpModel) {
                            if (httpModel.getSuccess() == 1) {
                                LanguageAdapter.this.changeLanguage(((LanguageModel) LanguageAdapter.this.mList.get(i)).getIso());
                            }
                        }
                    });
                }
            });
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public MyViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
            return new MyViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_language, viewGroup, false));
        }
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mRecyclerView.addItemDecoration(new DividerItemDecoration(this, 1));
        this.mRecyclerView.setAdapter(new LanguageAdapter(this, LanguageUtil.getLanguageList()));
    }

    private void initView() {
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mRecyclerView = (RecyclerView) findViewById(R.id.language_recyclerView);
        LanguageUtil.setText(this, this.mTitleTxtName, "changeLanguage");
    }

    private void setListener() {
        this.mTitleBack.setOnClickListener(this);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.title_img_back) {
            finish();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_language);
        initView();
        init();
        setListener();
    }
}
