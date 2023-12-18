#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <thread>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>  
#include <json11.hpp>
#include <curl/curl.h>
#include <inotify-cpp/NotifierBuilder.h>
#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP

el::Logger* defaultLogger = el::Loggers::getLogger("default");

class InstanceRegistry {
public:
    static InstanceRegistry& getInstance() {
        static InstanceRegistry instance;
        return instance;
    }

    std::string get_body_ip(const std::string& instance_name) {
        return body_instance_ip[instance_name];
    }

    std::string get_shadow_ip(const std::string& instance_name) {
        return shadow_instance_ip[instance_name];
    }

    std::string get_body_name(const std::string& shadow_name) {
        return shadow2body[shadow_name];
    }

    std::string get_shadow_name(const std::string& body_name) {
        return body2shadow[body_name];
    }

    void add_body(const std::string& instance_name, const std::string& ip) {
        body_instance_ip[instance_name] = ip;
    }

    void add_shadow(const std::string& instance_name, const std::string& ip, const std::string& body_name) {
        shadow_instance_ip[instance_name] = ip;
        body2shadow[body_name] = instance_name;
        shadow2body[instance_name] = body_name;
    }

    void delete_body(const std::string& instance_name) {
        body_instance_ip.erase(instance_name);
        shadow2body.erase(body2shadow[instance_name]);
        body2shadow.erase(instance_name);
    }

    std::string delete_shadow(const std::string& instance_name) {
        std::string body_name = shadow2body[instance_name];
        shadow_instance_ip.erase(instance_name);
        body2shadow.erase(body_name);
        shadow2body.erase(instance_name);
        return body_name;
    }

private:
    InstanceRegistry() = default;
    ~InstanceRegistry ()= default;
    InstanceRegistry(const InstanceRegistry&) = delete;
    InstanceRegistry& operator=(const InstanceRegistry&) = delete;

    std::map<std::string, std::string> body_instance_ip;
    std::map<std::string, std::string> shadow_instance_ip;
    std::map<std::string, std::string> body2shadow;
    std::map<std::string, std::string> shadow2body;
};

class MyHandler {
/* The on_closed function is triggered to handle file close_write event.
For AsyFunc, the filename prefix specifies the event type:
/*
+-------------------------------------+-----------------------------+-------------------------------------------------------+
| Prefix                              | Type                        | File Content Format                                   |
+-------------------------------------+-----------------------------+-------------------------------------------------------+
| AsyFunc-Registry-<instance name>    | Instance Registry.          | "type": {"body"|"shadow"}, "name": <instance name>,   |
|                                     |                             | "IP": <ip address>,                                   |
|                                     |                             | ["body": <paired body instance name>],                |
|                                     |                             | ["layers": <offload layers>]                          |
+-------------------------------------+-----------------------------+-------------------------------------------------------+
| AsyFunc-Delete-<instance name>      | Instance Delete.            | "type": {"body"|"shadow"}, "name": <instance name>    |
+-------------------------------------+-----------------------------+-------------------------------------------------------+
| AsyFunc-Body-<instance name> /      | Instance Communication.     | tensor data                                           |
| AsyFunc-Shadow-<instance name>      |                             |                                                       |
+-------------------------------------+-----------------------------+-------------------------------------------------------+
*/
public:
    MyHandler() {}
    void on_closed(std::filesystem::path path);
private:
    void update_layers(const std::string& body_name, const json11::Json& layers);
    void send_info(const std::string& ip, int port, const std::string& info);
};

void MyHandler::on_closed(std::filesystem::path path) {
    InstanceRegistry& registry = InstanceRegistry::getInstance();
    
    std::string filename = path.filename().string();
    if (filename.rfind("AsyFunc", 0) == std::string::npos) {
        return;
    }

    // handle instance registry event
    if (filename.rfind("AsyFunc-Registry", 0) != std::string::npos) {
        std::ifstream file(path);
        if (file.is_open()) {
            std::string fileContent((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            file.close();
            std::string err;
            json11::Json params = json11::Json::parse(fileContent, err);
            if (!err.empty()) {
                defaultLogger->error("Failed to parse JSON: %v", err);
                return;
            }
            std::string instance_type = params["type"].string_value();
            std::string instance_name = params["name"].string_value();
            std::string instance_ip = params["IP"].string_value();

            if (instance_type == "body") {
                registry.add_body(instance_name, instance_ip);
                defaultLogger->info("Registry Body instance: %v %v", instance_name, instance_ip);
            } else if (instance_type == "shadow") {
                std::string body_name = params["body"].string_value();
                registry.add_shadow(instance_name, instance_ip, body_name);

                json11::Json layers = params["layers"].is_array() ? params["layers"].array_items() : json11::Json::array{};
                defaultLogger->info("Registry Shadow instance: %v %v for %v with layers %v", instance_name, instance_ip, body_name, layers.dump());
                // Update the offload layers of the corresponding body function
                update_layers(body_name, layers);
            } else {
                defaultLogger->error("Error instance type: %v", instance_type);
            }
        }
        return;
    }
    // handle instance delete event
    else if (filename.rfind("AsyFunc-Delete", 0) != std::string::npos) {
        std::ifstream file(path);
        if (file.is_open()) {
            std::string fileContent((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            file.close();
            std::string err;
            json11::Json params = json11::Json::parse(fileContent, err);
            if (!err.empty()) {
                defaultLogger->fatal("Failed to parse JSON: %v", err);
                return;
            }
            std::string instance_type = params["type"].string_value();
            std::string instance_name = params["name"].string_value();

            defaultLogger->info("Delete instance: %v %v", instance_type, instance_name);

            if (instance_type == "body") {
                registry.delete_body(instance_name);
            } else if (instance_type == "shadow") {
                std::string body_name = registry.delete_shadow(instance_name);
                if (!body_name.empty()) {
                    // Update the offload layers of the corresponding body function to be empty
                    update_layers(body_name, json11::Json::array{});
                }
            } else {
                defaultLogger->error("Error instance type: %v", instance_type);
            }
        }
        return;
    }

    // handle instance communication event

    std::stringstream ss(filename);
    std::string field;
    std::vector<std::string> fields;

    while (std::getline(ss, field, '-')) {
        fields.push_back(field);
    }
    if (fields.size() < 3) {
        defaultLogger->error("Error filename: %v", filename);
        return;
    }
    std::string instance_name = fields[2];

    if (filename.rfind("AsyFunc-Body", 0) != std::string::npos) {
        std::string shadow_name = registry.get_shadow_name(instance_name);
        std::string ip = registry.get_shadow_ip(shadow_name);
        defaultLogger->info("Data offload: %v from %v to %v(%v)", path, instance_name, shadow_name, ip);
        send_info(ip, 8899, path);
    } else if (filename.rfind("AsyFunc-Shadow", 0) != std::string::npos) {
        std::string body_name = registry.get_body_name(instance_name);
        std::string ip = registry.get_body_ip(body_name);
        defaultLogger->info("Data offload: %v from %v to %v(%v)", path, instance_name, body_name, ip);
        send_info(ip, 8898, path);
    }  else {
        return;
    }
}


void MyHandler::update_layers(const std::string& body_name, const json11::Json &layers) {
    if (body_name.empty()) {
        return;
    }
    InstanceRegistry& registry = InstanceRegistry::getInstance();

    CURL* curl = curl_easy_init();
    if (curl) {
        std::string url = "http://" + registry.get_body_ip(body_name) + ":8080";
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        std::string postData = "{\"update\": true, \"layers\": " + layers.dump() + "}";
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postData.c_str());
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 2000L);
        CURLcode res = curl_easy_perform(curl);
        if(res != CURLE_OK) {
            defaultLogger->fatal("update_layers failed with url %v : %v", url, curl_easy_strerror(res));
        }
        curl_easy_cleanup(curl);
    }
}


void MyHandler::send_info(const std::string& ip, int port, const std::string& info) {
    int sk = socket(AF_INET, SOCK_STREAM, 0);
    if (sk == -1) {
        defaultLogger->fatal("Socket Creation Error");
        return;
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr.s_addr = inet_addr(ip.c_str());

    if (connect(sk, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        defaultLogger->fatal("Connect Server %v:%v Failed", ip, port);
        close(sk);
        return;
    }

    send(sk, info.c_str(), info.length(), 0);
    close(sk);
}

int main(int argc, char** argv)
{
    // set default log configurations
    el::Configurations defaultConf;
    defaultConf.setToDefault();
    defaultConf.set(el::Level::Global,
            el::ConfigurationType::Format, "%datetime %level --- %msg");
    el::Loggers::reconfigureLogger("default", defaultConf);

    // The path to be watched is passed as a command line argument
    std::string path = (argc > 1) ? argv[1] : ".";
    MyHandler handler;

    // Set the event handler which will be used to process particular events
    auto handleNotification = [&handler](inotify::Notification notification) {
        defaultLogger->info("Event %v on %v at %v was triggered.", notification.event, notification.path, 
                        notification.time.time_since_epoch().count());
        handler.on_closed(notification.path);
    };

    // Set the events to be notified for
    auto events = {inotify::Event::close_write};

    // The notifier is configured to watch the parsed path for the defined events.
    auto notifier = inotify::BuildNotifier()
                        .watchPathRecursively(path)
                        .onEvents(events, handleNotification);

    defaultLogger->info("*** Start Monitor Directory %v ***",  path);
    notifier.run();
    return 0;
}