#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"

// Zenoh-Pico v1.0.0+ API
#include "zenoh-pico.h"

#define WIFI_SSID "Fios-9ZCw6"
#define WIFI_PASS "deist69has49del"

static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                                    int32_t event_id, void* event_data) {
        if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
            esp_wifi_connect();
        } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
            printf("[Wi-Fi] Disconnected. Retrying connection...\n");
            esp_wifi_connect();
        } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
            ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
            printf("[Wi-Fi] Connected! Got IP: " IPSTR "\n", IP2STR(&event->ip_info.ip));
        }
    }

void connect_to_wifi(void) {
        // Create default Wi-Fi station
        esp_netif_create_default_wifi_sta();

        wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
        ESP_ERROR_CHECK(esp_wifi_init(&cfg));

        // Register event handlers for connection and IP assignment
        ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                            ESP_EVENT_ANY_ID,
                                                            &wifi_event_handler,
                                                            NULL,
                                                            NULL));
        ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                            IP_EVENT_STA_GOT_IP,
                                                            &wifi_event_handler,
                                                            NULL,
                                                            NULL));

        wifi_config_t wifi_config = {
            .sta = {
                .ssid = WIFI_SSID,
                .password = WIFI_PASS,
            },
        };
        
        ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
        ESP_ERROR_CHECK(esp_wifi_start());
        
        printf("[Wi-Fi] Initialization complete. Connecting to %s...\n", WIFI_SSID);
    }

void zenoh_task(void *pvParameters) {
    printf("Starting Zenoh Task...\n");

    // Wait until Wi-Fi connects and assigns an IP address
    // (You can also tie this to your Wi-Fi event handler flag)
    vTaskDelay(pdMS_TO_TICKS(4000));

    // 1. Initialize config (Default is Peer Mode)
    z_owned_config_t z_config;
    z_config_default(&z_config);

    // 2. Open session
    z_owned_session_t session;
    printf("Opening Zenoh Session...\n");
    if (z_open(&session, z_move(z_config), NULL) != 0) {
        printf("Failed to open Zenoh session\n");
        vTaskDelete(NULL);
    }
    printf("Zenoh session successfully opened!\n");
    
    // 3. Declare publisher
    z_owned_publisher_t pub;
    z_owned_keyexpr_t ke;
    z_keyexpr_from_str(&ke, "ross2/scout/1/ping");

    printf("Declaring Publisher...\n");
    if (z_declare_publisher(z_session_loan(&session), &pub, z_keyexpr_loan(&ke), NULL) != 0) {
        printf("Failed to declare publisher\n");
    }

    // 4. Setup put options
    z_publisher_put_options_t options;
    z_publisher_put_options_default(&options);

    // 5. Publish loop
    int count = 0;
    char payload[64];
    
    while (1) {
        int len = snprintf(payload, sizeof(payload), "Ping from Scout 1: %d", count++);
        
        z_owned_bytes_t payload_bytes;
        z_bytes_copy_from_buf(&payload_bytes, (const uint8_t*)payload, len); 
        
        if (z_publisher_put(z_publisher_loan(&pub), z_move(payload_bytes), &options) == 0) {
            printf("[Zenoh] Successfully published: %s\n", payload);
        } else {
            printf("[Zenoh] Failed to publish payload.\n");
        }
        
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
void app_main(void) {
    // Initialize NVS (Required for Wi-Fi)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Initialize networking event loop
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    // Connect to Wi-Fi before starting Zenoh
    // (Insert your specific Wi-Fi connection logic here)
    printf("Connecting to Wi-Fi...\n");
    connect_to_wifi(); 

    // Start the Zenoh background thread (8192 bytes stack size is recommended for network tasks)
    xTaskCreate(zenoh_task, "zenoh_task", 8192, NULL, 5, NULL);
}