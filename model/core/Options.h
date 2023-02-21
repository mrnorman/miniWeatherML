
#pragma once

#include "main_header.h"

// The Options class creates a way for users to set key,value pairs in the coupler
// where the key is always a string, and the value can be different types

namespace core {

  class Options {
  public:

    struct Option {
      std::string key;
      void *      data;
      size_t      type_hash;
    };

    std::vector<Option> options;


    Options() {}
    ~Options() { finalize(); }


    Options( Options &&rhs) = default;
    Options &operator=( Options &&rhs) = default;
    Options( Options const &dm ) = delete;
    Options &operator=( Options const &dm ) = delete;


    void finalize() {
      yakl::fence();
      for (int i=0; i < options.size(); i++) {
        delete_generic(i);
      }
      options = std::vector<Option>();
    }


    void delete_generic(int id) {
      if      (options[id].type_hash == get_type_hash<short int>             ()) { delete_specific<short int>             (id); }
      else if (options[id].type_hash == get_type_hash<int>                   ()) { delete_specific<int>                   (id); }
      else if (options[id].type_hash == get_type_hash<long int>              ()) { delete_specific<long int>              (id); }
      else if (options[id].type_hash == get_type_hash<long long int>         ()) { delete_specific<long long int>         (id); }
      else if (options[id].type_hash == get_type_hash<unsigned short int>    ()) { delete_specific<unsigned short int>    (id); }
      else if (options[id].type_hash == get_type_hash<unsigned int>          ()) { delete_specific<unsigned int>          (id); }
      else if (options[id].type_hash == get_type_hash<unsigned long int>     ()) { delete_specific<unsigned long int>     (id); }
      else if (options[id].type_hash == get_type_hash<unsigned long long int>()) { delete_specific<unsigned long long int>(id); }
      else if (options[id].type_hash == get_type_hash<float>                 ()) { delete_specific<float>                 (id); }
      else if (options[id].type_hash == get_type_hash<double>                ()) { delete_specific<double>                (id); }
      else if (options[id].type_hash == get_type_hash<long double>           ()) { delete_specific<long double>           (id); }
      else if (options[id].type_hash == get_type_hash<bool>                  ()) { delete_specific<bool>                  (id); }
      else if (options[id].type_hash == get_type_hash<char>                  ()) { delete_specific<char>                  (id); }
      else if (options[id].type_hash == get_type_hash<std::string>           ()) { delete_specific<std::string>           (id); }
    }


    template <class T>
    void delete_specific(int id) {
      delete (T *) options[id].data;
    }


    template <class T>
    void add_option( std::string key , T value ) {
      validate_type<T>();
      if ( key == "" ) return;
      int id = find_option( key );
      if ( id == -1 ) {
        T * ptr = new T(value);
        options.push_back({ key , (void *) ptr , get_type_hash<T>() });
      } else {
        *((T *) options[id].data) = value;
      }
    }


    template <class T>
    void set_option( std::string key , T value ) {
      validate_type<T>();
      if ( key == "" ) return;
      add_option( key , value );
    }


    template <class T>
    T get_option( std::string key ) const {
      validate_type<T>();
      int id = find_option_or_die( key );
      if (get_type_hash<T>() != options[id].type_hash) {
        std::cerr << "ERROR: Requesting option using the wrong type for key [" << key << "]" << std::endl;
        endrun("");
      }
      return *( (T *) options[id].data);
    }


    int find_option( std::string key ) const {
      for (int i=0; i < options.size(); i++) {
        if (key == options[i].key) return i;
      }
      return -1;
    }


    int find_option_or_die( std::string key ) const {
      int id = find_option(key);
      if (id >= 0) return id;
      std::cerr << "ERROR: Option not found for key [" << key << "]" << std::endl;
      endrun("");
      return -1;
    }


    bool option_exists( std::string key ) const {
      return find_option(key) >= 0;
    }


    int get_num_options() const {
      return options.size();
    }


    void delete_option( std::string key ) {
      int id = find_option(key);
      if (id >= 0) {
        delete_generic(id);
        options.erase( options.begin() + id );
      }
    }


    // INTERNAL USE: Return the C++ hash of this type. Ignore const and volatiles modifiers
    template <class T> size_t get_type_hash() const {
      return typeid(typename std::remove_cv<T>::type).hash_code();
    }


    template <class T>
    bool type_supported() const {
      if ( get_type_hash<T>() == get_type_hash<short int>             () ||
           get_type_hash<T>() == get_type_hash<int>                   () ||
           get_type_hash<T>() == get_type_hash<long int>              () ||
           get_type_hash<T>() == get_type_hash<long long int>         () ||
           get_type_hash<T>() == get_type_hash<unsigned short int>    () ||
           get_type_hash<T>() == get_type_hash<unsigned int>          () ||
           get_type_hash<T>() == get_type_hash<unsigned long int>     () ||
           get_type_hash<T>() == get_type_hash<unsigned long long int>() ||
           get_type_hash<T>() == get_type_hash<float>                 () ||
           get_type_hash<T>() == get_type_hash<double>                () ||
           get_type_hash<T>() == get_type_hash<long double>           () ||
           get_type_hash<T>() == get_type_hash<bool>                  () ||
           get_type_hash<T>() == get_type_hash<char>                  () ||
           get_type_hash<T>() == get_type_hash<std::string>           () ) return true;
      return false;
    }


    template <class T>
    void validate_type() const {
      if (! type_supported<T>() ) endrun("ERROR: Options type is not supported");
    }

  };

}


