package com.hismith.hismithapp;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.media3.extractor.text.ttml.TtmlNode;
import androidx.recyclerview.widget.DividerItemDecoration;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.CountryModel;
import com.hismith.hismithapp.model.Node;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class CountryActivity extends BaseActivity implements View.OnClickListener {
    private RetrofitRequest baseRequest;
    private String cityName;
    private String countryName;
    private LinearLayoutManager mLayoutManager;
    private RecyclerView mRecyclerView;
    private ImageView mTitleBack;
    private TextView mTitleName;
    private TextView mTxtAll;
    private TextView mTxtCountry;
    private TextView mTxtTarget;
    private int countryId = 0;
    private int cityId = 0;

    class CountryAdapter extends RecyclerView.Adapter {
        private List<Node> cityShownList;
        private Node currentChosenCity;
        private List<Node> mCitys;
        private Context mContext;
        private List<Integer> shrinkCityIdList = new ArrayList();

        class MyViewHolder extends RecyclerView.ViewHolder {
            private ImageView mImgConfirm;
            private ImageView mImgCountry;
            private LinearLayout mLayoutCountry;
            private TextView mTxtCountry;

            public MyViewHolder(View view) {
                super(view);
                this.mLayoutCountry = (LinearLayout) view.findViewById(R.id.item_layout_country);
                this.mImgCountry = (ImageView) view.findViewById(R.id.item_img_country);
                this.mImgConfirm = (ImageView) view.findViewById(R.id.item_img_country_confirm);
                this.mTxtCountry = (TextView) view.findViewById(R.id.item_txt_country);
            }
        }

        public CountryAdapter(Context context, List<Node> list, int i) {
            this.mContext = context;
            this.mCitys = list;
            this.cityShownList = new ArrayList(i);
            for (int i2 = 0; i2 < list.size(); i2++) {
                if (list.get(i2).isHasChild()) {
                    this.shrinkCityIdList.add(list.get(i2).getId());
                }
            }
            if (list != null && list.size() > 0) {
                this.currentChosenCity = list.get(0);
            }
            initCustomerShownList();
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public int getItemCount() {
            return this.cityShownList.size();
        }

        public void initCustomerShownList() {
            this.cityShownList.clear();
            for (int i = 0; i < this.mCitys.size(); i++) {
                Node node = this.mCitys.get(i);
                if (!this.shrinkCityIdList.contains(node.getPid())) {
                    this.cityShownList.add(node);
                }
            }
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public void onBindViewHolder(RecyclerView.ViewHolder viewHolder, final int i) {
            MyViewHolder myViewHolder = (MyViewHolder) viewHolder;
            myViewHolder.mTxtCountry.setText(this.cityShownList.get(i).getName());
            if (this.cityShownList.get(i).isHasChild()) {
                if (this.shrinkCityIdList.contains(this.cityShownList.get(i).getId())) {
                    myViewHolder.mImgCountry.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_personal_arrow));
                } else {
                    myViewHolder.mImgCountry.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_personal_arrow_down));
                }
                myViewHolder.mImgCountry.setVisibility(0);
            } else {
                myViewHolder.mImgCountry.setVisibility(8);
            }
            myViewHolder.mImgConfirm.setVisibility(8);
            int intValue = this.cityShownList.get(i).getPid().intValue();
            CountryActivity countryActivity = CountryActivity.this;
            if (intValue == 0) {
                if (countryActivity.countryId == this.cityShownList.get(i).getId().intValue() && !this.cityShownList.get(i).isHasChild()) {
                    myViewHolder.mImgCountry.setVisibility(8);
                    myViewHolder.mImgConfirm.setVisibility(0);
                }
            } else if (countryActivity.countryId == this.cityShownList.get(i).getPid().intValue() && CountryActivity.this.cityId == this.cityShownList.get(i).getId().intValue()) {
                myViewHolder.mImgCountry.setVisibility(8);
                myViewHolder.mImgConfirm.setVisibility(0);
            }
            LinearLayout.LayoutParams layoutParams = (LinearLayout.LayoutParams) myViewHolder.mTxtCountry.getLayoutParams();
            layoutParams.setMargins(MeasureUtil.dip2px(CountryActivity.this, 20.0f) * this.cityShownList.get(i).getLevel().intValue(), 0, 0, 0);
            myViewHolder.mTxtCountry.setLayoutParams(layoutParams);
            myViewHolder.mLayoutCountry.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.CountryActivity.CountryAdapter.1
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    boolean isHasChild = ((Node) CountryAdapter.this.cityShownList.get(i)).isHasChild();
                    CountryAdapter countryAdapter = CountryAdapter.this;
                    if (isHasChild) {
                        boolean contains = countryAdapter.shrinkCityIdList.contains(((Node) CountryAdapter.this.cityShownList.get(i)).getId());
                        CountryAdapter countryAdapter2 = CountryAdapter.this;
                        if (contains) {
                            countryAdapter2.shrinkCityIdList.remove(((Node) CountryAdapter.this.cityShownList.get(i)).getId());
                        } else {
                            countryAdapter2.shrinkCityIdList.add(((Node) CountryAdapter.this.cityShownList.get(i)).getId());
                        }
                        CountryAdapter.this.initCustomerShownList();
                        CountryAdapter.this.notifyDataSetChanged();
                        return;
                    }
                    countryAdapter.currentChosenCity = (Node) countryAdapter.cityShownList.get(i);
                    int intValue2 = CountryAdapter.this.currentChosenCity.getPid().intValue();
                    CountryAdapter countryAdapter3 = CountryAdapter.this;
                    if (intValue2 == 0) {
                        CountryActivity.this.countryId = CountryAdapter.this.currentChosenCity.getId().intValue();
                        CountryActivity.this.countryName = CountryAdapter.this.currentChosenCity.getName();
                        CountryActivity.this.cityId = 0;
                        CountryActivity.this.cityName = "";
                    } else {
                        CountryActivity.this.countryId = CountryAdapter.this.currentChosenCity.getPid().intValue();
                        CountryActivity.this.countryName = CountryAdapter.this.currentChosenCity.getpName();
                        CountryActivity.this.cityId = CountryAdapter.this.currentChosenCity.getId().intValue();
                        CountryActivity.this.cityName = CountryAdapter.this.currentChosenCity.getName();
                    }
                    CountryActivity.this.initCurrCountry();
                    Intent intent = new Intent();
                    intent.putExtra(Constans.resultCountryId, CountryActivity.this.countryId);
                    intent.putExtra(Constans.resultCountryName, CountryActivity.this.countryName);
                    intent.putExtra(Constans.resultCityId, CountryActivity.this.cityId);
                    intent.putExtra(Constans.resultCityName, CountryActivity.this.cityName);
                    CountryActivity.this.setResult(11, intent);
                    CountryActivity.this.appManager.finishActivity();
                }
            });
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public RecyclerView.ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
            return new MyViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_country, viewGroup, false));
        }
    }

    private void getCountry() {
        this.baseRequest.getApi().getCountry().subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<CountryModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.CountryActivity.1
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(CountryModel countryModel) {
                if (countryModel.getSuccess() != 1 || countryModel.getCountrys() == null) {
                    return;
                }
                List initCountryData = CountryActivity.this.initCountryData(countryModel.getCountrys(), 0, "", 0);
                CountryActivity countryActivity = CountryActivity.this;
                CountryActivity.this.mRecyclerView.setAdapter(countryActivity.new CountryAdapter(countryActivity, initCountryData, countryModel.getCountrys().length));
            }
        });
    }

    private void init() {
        if (getIntent() != null) {
            this.countryId = getIntent().getIntExtra(Constans.resultCountryId, 0);
            this.cityId = getIntent().getIntExtra(Constans.resultCityId, 0);
            this.countryName = getIntent().getStringExtra(Constans.resultCountryName);
            this.cityName = getIntent().getStringExtra(Constans.resultCityName);
        }
        LogUtil.e("countryId:" + this.countryId + ", cityId:" + this.cityId);
        initCurrCountry();
        this.baseRequest = new RetrofitRequest(RetrofitRequest.EnumRequest.REQUEST_BASE);
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mRecyclerView.addItemDecoration(new DividerItemDecoration(this, 1));
        getCountry();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public List<Node> initCountryData(CountryModel.City[] cityArr, int i, String str, int i2) {
        ArrayList arrayList = new ArrayList();
        int i3 = 0;
        while (i3 < cityArr.length) {
            String str2 = str;
            Node node = new Node(cityArr[i3].getId(), cityArr[i3].getName(), Integer.valueOf(i), str2, Integer.valueOf(i2));
            if (cityArr[i3].getCitys() != null && cityArr[i3].getCitys().length > 0) {
                node.setHasChild(true);
            }
            arrayList.add(node);
            if (node.isHasChild()) {
                arrayList.addAll(initCountryData(cityArr[i3].getCitys(), cityArr[i3].getId().intValue(), cityArr[i3].getName(), i2 + 1));
            }
            i3++;
            str = str2;
        }
        return arrayList;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initCurrCountry() {
        String str = this.countryName;
        if (str == null || str.length() <= 0) {
            return;
        }
        String str2 = this.cityName;
        this.mTxtCountry.setText((str2 == null || str2.length() <= 0) ? this.countryName : this.countryName + ", " + this.cityName);
    }

    private void initView() {
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleName = (TextView) findViewById(R.id.title_txt_name);
        this.mRecyclerView = (RecyclerView) findViewById(R.id.country_recycler);
        this.mTxtTarget = (TextView) findViewById(R.id.txt_target);
        this.mTxtAll = (TextView) findViewById(R.id.txt_all);
        this.mTxtCountry = (TextView) findViewById(R.id.txt_country);
        LanguageUtil.setText(this, this.mTitleName, "pleaseSelectCountry");
        LanguageUtil.setText(this, this.mTxtTarget, "targetedArea");
        LanguageUtil.setText(this, this.mTxtAll, TtmlNode.COMBINE_ALL);
    }

    private void setListener() {
        this.mTitleBack.setOnClickListener(this);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.title_img_back) {
            this.appManager.finishActivity();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_country);
        initView();
        init();
        setListener();
    }
}
