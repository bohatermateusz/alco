package com.hismith.hismithapp.twitter;

import android.content.Intent;
import android.util.Base64;
import com.facebook.internal.ServerProtocol;
import com.hismith.hismithapp.LoginActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.WebActivity;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.ihsanbal.logging.Level;
import com.ihsanbal.logging.LoggingInterceptor;
import com.twitter.sdk.android.core.internal.oauth.OAuthConstants;
import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.TreeMap;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.FormBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import org.json.JSONException;
import org.json.JSONObject;

/* loaded from: classes3.dex */
public class OAuthSigner {
    private OkHttpClient client;
    private LoginActivity mContext;
    private int requestTwitter = 333;
    private String bearerToken = "AAAAAAAAAAAAAAAAAAAAAL79%2FgAAAAAAhmxZfNxQ5YcUTaw64FIon1Q8Wq8%3DzbGhrY1tGNPPxPaWizHBIASQSSYMMixHJ8iNMfQU8k0l2D96oZ";

    public interface OnOtherLoginListener {
        void onOtherLogin(String str);
    }

    public OAuthSigner(LoginActivity loginActivity) {
        this.mContext = loginActivity;
        OkHttpClient.Builder builder = new OkHttpClient.Builder();
        builder.addInterceptor(new LoggingInterceptor.Builder().loggable(true).setLevel(Level.BASIC).log(5).request("request").response("response").build());
        this.client = builder.build();
    }

    public static String encode(String str) {
        return URLEncoder.encode(str, StandardCharsets.UTF_8).replace("+", "%20").replace("*", "%2A").replace("%7E", "~");
    }

    private String generateOAuthNonce() {
        byte[] bArr = new byte[32];
        new SecureRandom().nextBytes(bArr);
        return Base64.encodeToString(bArr, 2);
    }

    public static String generateSignature(String str, String str2, String str3, String str4, String str5, String str6, String str7, String str8, String str9) throws Exception {
        TreeMap treeMap = new TreeMap();
        treeMap.put(OAuthConstants.PARAM_CONSUMER_KEY, str3);
        treeMap.put(OAuthConstants.PARAM_NONCE, str7);
        treeMap.put(OAuthConstants.PARAM_SIGNATURE_METHOD, "HMAC-SHA1");
        treeMap.put(OAuthConstants.PARAM_TIMESTAMP, str8);
        treeMap.put(OAuthConstants.PARAM_VERSION, "1.0");
        if (str5 != null) {
            treeMap.put(OAuthConstants.PARAM_TOKEN, str5);
        }
        if (str9 != null) {
            treeMap.put(OAuthConstants.PARAM_CALLBACK, str9);
        }
        StringBuilder sb = new StringBuilder();
        for (String str10 : treeMap.keySet()) {
            if (sb.length() > 0) {
                sb.append("&");
            }
            sb.append(encode(str10)).append("=").append(encode((String) treeMap.get(str10)));
        }
        return encode(Base64.encodeToString(hmacSha1(encode(str4) + "&" + (str6 != null ? encode(str6) : ""), str.toUpperCase() + "&" + encode(str2) + "&" + encode(sb.toString())), 2));
    }

    private String getTimestampSeconds() {
        return String.valueOf(System.currentTimeMillis() / 1000);
    }

    private static byte[] hmacSha1(String str, String str2) throws NoSuchAlgorithmException, InvalidKeyException {
        SecretKeySpec secretKeySpec = new SecretKeySpec(str.getBytes(StandardCharsets.UTF_8), "HmacSHA1");
        Mac mac = Mac.getInstance("HmacSHA1");
        mac.init(secretKeySpec);
        return mac.doFinal(str2.getBytes(StandardCharsets.UTF_8));
    }

    private void requestAccessToken() {
        this.client.newCall(new Request.Builder().url("https://api.x.com/oauth2/token").addHeader("Authorization", "Basic " + Base64.encodeToString((this.mContext.getString(R.string.twitter_consumer_key) + ":" + this.mContext.getString(R.string.twitter_consumer_secret)).getBytes(), 2)).post(new FormBody.Builder().add(OAuthConstants.PARAM_GRANT_TYPE, OAuthConstants.GRANT_TYPE_CLIENT_CREDENTIALS).build()).build()).enqueue(new Callback() { // from class: com.hismith.hismithapp.twitter.OAuthSigner.2
            @Override // okhttp3.Callback
            public void onFailure(Call call, IOException iOException) {
                LogUtil.e("twitter requestAccessToken onFailure " + iOException.getMessage());
                iOException.printStackTrace();
            }

            @Override // okhttp3.Callback
            public void onResponse(Call call, Response response) throws IOException {
                if (!response.isSuccessful()) {
                    LogUtil.e("twitter requestAccessToken Unexpected code " + response);
                    throw new IOException("Unexpected code " + response);
                }
                String string = response.body().string();
                LogUtil.e("twitter requestAccessToken success:" + string);
                try {
                    String string2 = new JSONObject(string).getString("access_token");
                    LogUtil.e("twitter requestAccessToken access_token:" + string2);
                    OAuthSigner.this.bearerToken = string2;
                } catch (JSONException e) {
                    throw new RuntimeException(e);
                }
            }
        });
    }

    private void requestInvalidateToken(String str) {
        this.client.newCall(new Request.Builder().url("https://api.x.com/oauth2/invalidate_token").addHeader("authorization", "OAuth oauth_consumer_key=\"" + this.mContext.getString(R.string.twitter_consumer_key) + "\",oauth_nonce=\"AUTO_GENERATED_NONCE\", oauth_signature=\"AUTO_GENERATED_SIGNATURE\",oauth_signature_method=\"HMAC-SHA1\", oauth_timestamp=\"AUTO_GENERATED_TIMESTAMP\",oauth_token=\"ACCESS_TOKEN\", oauth_version=\"1.0\"").post(new FormBody.Builder().add("access_token", str).build()).build()).enqueue(new Callback() { // from class: com.hismith.hismithapp.twitter.OAuthSigner.3
            @Override // okhttp3.Callback
            public void onFailure(Call call, IOException iOException) {
                LogUtil.e("twitter requestInvalidateToken onFailure " + iOException.getMessage());
                iOException.printStackTrace();
            }

            @Override // okhttp3.Callback
            public void onResponse(Call call, Response response) throws IOException {
                if (!response.isSuccessful()) {
                    LogUtil.e("twitter requestInvalidateToken Unexpected code " + response);
                    throw new IOException("Unexpected code " + response);
                }
                LogUtil.e("twitter requestInvalidateToken success:" + response.body().string());
            }
        });
    }

    private void requestOauth2Authorize() {
        String str = "https://twitter.com/i/oauth2/authorize?response_type=code&client_id=ZGdTajVVOUJXc1J1TzhaZF8wbzQ6MTpjaQ&redirect_uri=twittersdk://hismith&scope=users.read tweet.read&state=" + String.valueOf(System.currentTimeMillis()) + "&code_challenge_method=S256";
        Intent intent = new Intent(this.mContext, (Class<?>) WebActivity.class);
        intent.putExtra("extra_web_url", str);
        this.mContext.startActivityForResult(intent, this.requestTwitter);
    }

    private void requestToken1() {
        try {
            String generateOAuthNonce = generateOAuthNonce();
            String valueOf = String.valueOf(System.currentTimeMillis() / 1000);
            this.client.newCall(new Request.Builder().url("https://api.x.com/oauth/request_token").addHeader("authorization", String.format("OAuth oauth_consumer_key=\"%s\", oauth_nonce=\"%s\", oauth_signature=\"%s\", oauth_signature_method=\"HMAC-SHA1\", oauth_timestamp=\"%s\", oauth_version=\"1.0\", oauth_callback=\"%s\"", this.mContext.getString(R.string.twitter_consumer_key), generateOAuthNonce, encode(generateSignature("POST", "https://api.x.com/oauth/request_token", this.mContext.getString(R.string.twitter_consumer_key), this.mContext.getString(R.string.twitter_consumer_secret), null, null, generateOAuthNonce, valueOf, "twitterkit-DJoknwnzJMQcoMyz2is6UygMa://")), valueOf, "twitterkit-DJoknwnzJMQcoMyz2is6UygMa://")).post(new FormBody.Builder().build()).build()).enqueue(new Callback() { // from class: com.hismith.hismithapp.twitter.OAuthSigner.1
                @Override // okhttp3.Callback
                public void onFailure(Call call, IOException iOException) {
                    LogUtil.e("twitter request_token onFailure:" + iOException.getMessage());
                }

                @Override // okhttp3.Callback
                public void onResponse(Call call, Response response) throws IOException {
                    LogUtil.e("twitter request_token onResponse:" + response.toString());
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void tweedleOauth2() {
    }

    public void authTwitter() {
        requestAccessToken();
    }

    public void getOauth2AccessToken2(String str, String str2, final OnOtherLoginListener onOtherLoginListener) {
        this.client.newCall(new Request.Builder().url("https://api.x.com/2/oauth2/token").addHeader("Authorization", "Basic " + Base64.encodeToString("ZGdTajVVOUJXc1J1TzhaZF8wbzQ6MTpjaQ:FmYaoKhfWRF3tSeJHiR2DAEYJz-ihQlq2IJJC0AlU7-ZfLlhys".getBytes(), 2)).post(new FormBody.Builder().add("code", str).add(OAuthConstants.PARAM_GRANT_TYPE, "authorization_code").add("client_id", "ZGdTajVVOUJXc1J1TzhaZF8wbzQ6MTpjaQ").add(ServerProtocol.DIALOG_PARAM_REDIRECT_URI, Constans.twitter_redirectUri).add("code_verifier", str2).build()).build()).enqueue(new Callback() { // from class: com.hismith.hismithapp.twitter.OAuthSigner.4
            @Override // okhttp3.Callback
            public void onFailure(Call call, IOException iOException) {
                LogUtil.e("twitter token onFailure " + iOException.getMessage());
                iOException.printStackTrace();
            }

            @Override // okhttp3.Callback
            public void onResponse(Call call, Response response) throws IOException {
                if (!response.isSuccessful()) {
                    LogUtil.e("twitter token Unexpected code " + response);
                    throw new IOException("Unexpected code " + response);
                }
                String string = response.body().string();
                LogUtil.e("twitter token success:" + string);
                try {
                    String string2 = new JSONObject(string).getString("access_token");
                    LogUtil.e("twitter token access_token:" + string2);
                    OnOtherLoginListener onOtherLoginListener2 = onOtherLoginListener;
                    if (onOtherLoginListener2 != null) {
                        onOtherLoginListener2.onOtherLogin(string2);
                    }
                } catch (JSONException e) {
                    throw new RuntimeException(e);
                }
            }
        });
    }

    public void getOauth2UserLookupMe(String str) {
        String str2 = "Basic " + Base64.encodeToString("ZGdTajVVOUJXc1J1TzhaZF8wbzQ6MTpjaQ:FmYaoKhfWRF3tSeJHiR2DAEYJz-ihQlq2IJJC0AlU7-ZfLlhys".getBytes(), 2);
        this.client.newCall(new Request.Builder().url("https://api.twitter.com/2/users/me").addHeader("Authorization", "Bearer " + str).build()).enqueue(new Callback() { // from class: com.hismith.hismithapp.twitter.OAuthSigner.5
            @Override // okhttp3.Callback
            public void onFailure(Call call, IOException iOException) {
                LogUtil.e("twitter UserLookupMe onFailure " + iOException.getMessage());
                iOException.printStackTrace();
            }

            @Override // okhttp3.Callback
            public void onResponse(Call call, Response response) throws IOException {
                if (!response.isSuccessful()) {
                    LogUtil.e("twitter UserLookupMe Unexpected code " + response);
                    throw new IOException("Unexpected code " + response);
                }
                String string = response.body().string();
                LogUtil.e("twitter UserLookupMe success:" + string);
                try {
                    new JSONObject(string);
                } catch (JSONException e) {
                    throw new RuntimeException(e);
                }
            }
        });
    }
}
