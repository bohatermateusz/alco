package com.hismith.hismithapp.http;

import com.google.common.net.HttpHeaders;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.ihsanbal.logging.Level;
import com.ihsanbal.logging.LoggingInterceptor;
import java.io.IOException;
import java.lang.annotation.Annotation;
import java.lang.reflect.Type;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.concurrent.TimeUnit;
import javax.net.ssl.X509TrustManager;
import okhttp3.Interceptor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.ResponseBody;
import retrofit2.Converter;
import retrofit2.Retrofit;
import retrofit2.adapter.rxjava2.RxJava2CallAdapterFactory;
import retrofit2.converter.gson.GsonConverterFactory;

/* loaded from: classes3.dex */
public class RetrofitRequest {
    private HismithApi api;
    private Retrofit retrofit;
    private final int TIME_OUT = 60;
    private boolean isDeBug = false;
    final X509TrustManager trustAllCert = new X509TrustManager() { // from class: com.hismith.hismithapp.http.RetrofitRequest.3
        @Override // javax.net.ssl.X509TrustManager
        public void checkClientTrusted(X509Certificate[] x509CertificateArr, String str) throws CertificateException {
        }

        @Override // javax.net.ssl.X509TrustManager
        public void checkServerTrusted(X509Certificate[] x509CertificateArr, String str) throws CertificateException {
        }

        @Override // javax.net.ssl.X509TrustManager
        public X509Certificate[] getAcceptedIssuers() {
            return new X509Certificate[0];
        }
    };

    public enum EnumRequest {
        REQUEST_BASE,
        REQUEST_FILE
    }

    class NullOnEmptyConverterFactory extends Converter.Factory {
        NullOnEmptyConverterFactory() {
        }

        @Override // retrofit2.Converter.Factory
        public Converter<ResponseBody, ?> responseBodyConverter(Type type, Annotation[] annotationArr, Retrofit retrofit) {
            final Converter nextResponseBodyConverter = retrofit.nextResponseBodyConverter(this, type, annotationArr);
            return new Converter<ResponseBody, Object>() { // from class: com.hismith.hismithapp.http.RetrofitRequest.NullOnEmptyConverterFactory.1
                @Override // retrofit2.Converter
                public Object convert(ResponseBody responseBody) throws IOException {
                    if (responseBody.contentLength() == 0) {
                        return null;
                    }
                    return nextResponseBodyConverter.convert(responseBody);
                }
            };
        }
    }

    public RetrofitRequest() {
        initBaseRetrofit();
    }

    public RetrofitRequest(EnumRequest enumRequest) {
        int ordinal = enumRequest.ordinal();
        if (ordinal == 0) {
            initBaseRetrofit();
        } else {
            if (ordinal != 1) {
                return;
            }
            initFileRetrofit();
        }
    }

    private Interceptor createFilteredLoggingInterceptor() {
        final LoggingInterceptor build = new LoggingInterceptor.Builder().loggable(this.isDeBug).setLevel(Level.BASIC).log(5).request("request").response("response").build();
        return new Interceptor() { // from class: com.hismith.hismithapp.http.RetrofitRequest.2
            @Override // okhttp3.Interceptor
            public Response intercept(Interceptor.Chain chain) throws IOException {
                Request request = chain.request();
                return request.url().toString().contains("messageNew_v1") ? chain.proceed(request) : build.intercept(chain);
            }
        };
    }

    private void initBaseRetrofit() {
        OkHttpClient.Builder builder = new OkHttpClient.Builder();
        builder.connectTimeout(60L, TimeUnit.SECONDS).readTimeout(60L, TimeUnit.SECONDS).writeTimeout(60L, TimeUnit.SECONDS);
        builder.sslSocketFactory(new SSLSocketFactoryCompat(this.trustAllCert), this.trustAllCert);
        builder.addInterceptor(new Interceptor() { // from class: com.hismith.hismithapp.http.RetrofitRequest.1
            @Override // okhttp3.Interceptor
            public Response intercept(Interceptor.Chain chain) throws IOException {
                Request request = chain.request();
                return chain.proceed(request.newBuilder().header(HttpHeaders.ACCEPT_LANGUAGE, LanguageUtil.getCurrLanguage(MyApplication.getApplication())).method(request.method(), request.body()).build());
            }
        });
        if (this.isDeBug) {
            builder.addInterceptor(createFilteredLoggingInterceptor());
        }
        Retrofit build = new Retrofit.Builder().client(builder.build()).baseUrl("https://www.hismiths.com/api/").addCallAdapterFactory(RxJava2CallAdapterFactory.create()).addConverterFactory(GsonConverterFactory.create()).build();
        this.retrofit = build;
        this.api = (HismithApi) build.create(HismithApi.class);
    }

    private void initFileRetrofit() {
        OkHttpClient.Builder builder = new OkHttpClient.Builder();
        builder.connectTimeout(60L, TimeUnit.SECONDS).readTimeout(60L, TimeUnit.SECONDS).writeTimeout(60L, TimeUnit.SECONDS);
        LogUtil.e("httpclient:connectTimeout:" + builder.build().connectTimeoutMillis() + ", writeTimeout:" + builder.build().writeTimeoutMillis() + ", readTimeout:" + builder.build().readTimeoutMillis());
        if (this.isDeBug) {
            builder.addInterceptor(createFilteredLoggingInterceptor());
        }
        Retrofit build = new Retrofit.Builder().client(builder.build()).baseUrl("https://www.hismiths.com/api/").addCallAdapterFactory(RxJava2CallAdapterFactory.create()).addConverterFactory(GsonConverterFactory.create()).build();
        this.retrofit = build;
        this.api = (HismithApi) build.create(HismithApi.class);
    }

    public HismithApi getApi() {
        return this.api;
    }
}
