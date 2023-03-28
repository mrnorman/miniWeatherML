
#pragma once

#include "main_header.h"
#include <typeinfo>

// The DataManager class is intended to hold arrays of varying dimensions and types so that a coupler can have all
// data of all types in one place. 
// The user is expected to use the following types that are typedef'd at the bottom:
//       typedef DataManagerTemplate<yakl::memDevice> DataManager;
//       typedef DataManagerTemplate<yakl::memHost> DataManagerHost;

namespace core {

  using yakl::Array;

  extern std::mutex data_manager_mutex;


  template <int memSpace = yakl::memDevice>
  class DataManagerTemplate {
  public:

    struct Entry {
      std::string              name;
      std::string              desc;
      size_t                   type_hash;
      void *                   ptr;
      size_t                   bytes;
      std::vector<int>         dims;
      std::vector<std::string> dim_names;
      bool                     positive;
      bool                     dirty;
    };

    struct Dimension {
      std::string name;
      int         len;
    };

    std::vector<Entry>     entries;
    std::vector<Dimension> dimensions;

    std::function<void *( size_t , char const * )> allocate;
    std::function<void  ( void * , char const * )> deallocate;

    int num_assigned_dims;


    DataManagerTemplate() {
      entries    = std::vector<Entry>();
      dimensions = std::vector<Dimension>();
      num_assigned_dims = 0;
      if (memSpace == memDevice) {
        allocate   = [] (size_t bytes,char const *label) -> void * { return yakl::alloc_device(bytes,label); };
        deallocate = [] (void *ptr   ,char const *label)           {        yakl::free_device (ptr  ,label); };
      } else if (memSpace == memHost) {
        allocate   = [] (size_t bytes,char const *label) -> void * { return ::malloc(bytes); };
        deallocate = [] (void *ptr   ,char const *label)           {        ::free  (ptr); };
      } else {
        yakl::yakl_throw("ERROR: DataManagerTemplate created with invalid memSpace template parameter");
      }
    }

    DataManagerTemplate( DataManagerTemplate &&rhs) = default;
    DataManagerTemplate &operator=( DataManagerTemplate &&rhs) = default;
    DataManagerTemplate( DataManagerTemplate const &dm ) = delete;
    DataManagerTemplate &operator=( DataManagerTemplate const &dm ) = delete;


    ~DataManagerTemplate() {
      // finalize deallocates all entries and resets entries and dimensions to empty vectors
      finalize();
    }


    
    // Clone this DataManager object into the given DataManager object
    void clone_into( DataManagerTemplate<memSpace> &dm ) {
      dm.allocate          = this->allocate;
      dm.deallocate        = this->deallocate;
      dm.dimensions        = this->dimensions;
      dm.num_assigned_dims = this->num_assigned_dims;
      for (auto &entry : this->entries) {
        Entry loc;
        loc.name      = entry.name;
        loc.desc      = entry.desc;
        loc.type_hash = entry.type_hash;
        loc.ptr       = allocate( entry.bytes , entry.name.c_str() );
        if (memSpace == yakl::memHost) {
          yakl::memcpy_host_to_host_void    ( loc.ptr , entry.ptr , entry.bytes );
        } else {
          yakl::memcpy_device_to_device_void( loc.ptr , entry.ptr , entry.bytes );
          yakl::fence();
        }
        loc.bytes     = entry.bytes;
        loc.dims      = entry.dims;
        loc.dim_names = entry.dim_names;
        loc.positive  = entry.positive;
        loc.dirty     = entry.dirty;
        dm.entries.push_back(loc);
      }
    }


    void add_dimension( std::string name , int len ) {
      int dimid = find_dimension( name );
      if (dimid > 0) {
        if ( dimensions[dimid].len != len ) {
          std::cerr << "ERROR: Attempting to add a dimension of name [" << name << "]. " <<
                       "However, it already exists with length [" << dimensions[dimid].len << "].";
          endrun("");
        }
        return;  // Avoid adding a duplicate entry
      }
      dimensions.push_back( {name , len} );
    }


    // Create an entry and allocate it. if dim_names is passed, then check dimension sizes for consistency
    // if positive == true, then positivity validation checks for positivity; otherwise, ignores it.
    // While zeroing the allocation upon creation might be nice, it's not efficient in all GPU contexts
    // because many separate kernels are more expensive than one big one when data sizes are small.
    // So it's up to the user to zero out the arrays they allocate to make valgrind happy and avoid unhappy
    // irreproducible bugs.
    template <class T>
    void register_and_allocate( std::string name ,
                                std::string desc ,
                                std::vector<int> dims ,
                                std::vector<std::string> dim_names = std::vector<std::string>() ,
                                bool positive = false ) {
      static std::mutex data_manager_mutex;
      if (name == "") {
        endrun("ERROR: You cannot register_and_allocate with an empty string");
      }
      // Make sure we don't have a duplicate entry
      if ( find_entry(name) != -1) {
        std::cerr << "ERROR: Trying to register and allocate name [" << name << "], which already exists";
        endrun("");
      }

      if (dim_names.size() > 0) {
        if (dims.size() != dim_names.size()) {
          std::cerr << "ERROR: Trying to register and allocate name [" << name << "]. ";
          endrun("Must have the same number of dims and dim_names");
        }
        // Make sure the dimensions are the same size as existing ones of the same name
        for (int i=0; i < dim_names.size(); i++) {
          int dimid = find_dimension(dim_names[i]);
          if (dimid == -1) {
            Dimension loc;
            loc.name = dim_names[i];
            loc.len  = dims     [i];
            dimensions.push_back(loc);
          } else {
            if (dimensions[dimid].len != dims[i]) {
              std::cerr << "ERROR: Trying to register and allocate name [" << name << "]. " <<
                           "Dimension of name [" << dim_names[i] << "] already exists with a different " <<
                           "length of [" << dimensions[dimid].len << "]. The length you provided for " << 
                           "that dimension name in this call is [" << dims[i] << "]. ";
              endrun("");
            }
          }
        }
      } else {
        // If dim_names was not passed, then let's try to find some. If we can't, then we'll have to create them
        std::string loc_dim_name = "";
        for (int i=0; i < dims.size(); i++) { // i is local var dims index
          for (int ii=0; ii < this->dimensions.size(); ii++) { // ii is data manager dimensions index
            if (dims[i] == this->dimensions[ii].len) loc_dim_name = this->dimensions[i].name;
            break;
          }
          if (loc_dim_name == "") {
            data_manager_mutex.lock();
            loc_dim_name = std::string("assigned_dim_") + std::to_string(this->num_assigned_dims);
            this->num_assigned_dims++;
            data_manager_mutex.unlock();
          }
          dim_names.push_back(loc_dim_name);
        }
      }

      Entry loc;
      loc.name      = name;
      loc.desc      = desc;
      loc.type_hash = get_type_hash<T>();
      loc.ptr       = allocate( get_data_size(dims)*sizeof(T) , name.c_str() );
      loc.bytes     = get_data_size(dims)*sizeof(T);
      loc.dims      = dims;
      loc.dim_names = dim_names;
      loc.positive  = positive;
      loc.dirty     = false;

      entries.push_back( loc );
    }


    // deallocate a named entry, and erase the entry from the list
    void unregister_and_deallocate( std::string name ) {
      int id = find_entry_or_error( name );
      deallocate( entries[id].ptr , entries[id].name.c_str() );
      entries.erase( entries.begin() + id );
    }


    // reset the dirty flag to false for all entries
    // when the dirty flag is true, then the entry has been potentially written to since its creation or previous cleaning
    void clean_all_entries() {
      for (int i=0; i < entries.size(); i++) { entries[i].dirty = false; }
    }


    // reset the dirty flag to false for a single entry
    // when the dirty flag is true, then the entry has been potentially written to since its creation or previous cleaning
    void clean_entry( std::string name ) {
      int id = find_entry_or_error( name );
      entries[id].dirty = false;
    }


    // Get the dirty flag for a single entry
    // when the dirty flag is true, then the entry has been potentially written to since its creation or previous cleaning
    bool entry_is_dirty( std::string name ) const {
      int id = find_entry_or_error( name );
      return entries[id].dirty;
    }


    // Get a list of entry names that are dirty
    // when the dirty flag is true, then the entry has been potentially written to since its creation or previous cleaning
    std::vector<std::string> get_dirty_entries( ) const {
      std::vector<std::string> dirty_entries;
      for (int i=0; i < entries.size(); i++) {
        if (entries[i].dirty) dirty_entries.push_back( entries[i].name );
      }
      return dirty_entries;
    }


    bool entry_exists( std::string name ) const {
      int id = find_entry(name);
      if (id >= 0) return true;
      return false;
    }


    // Get a READ ONLY YAKL array (styleC) for the entry of this name
    // If T is not const, then the dirty flag is set to true because it can be potentially written to
    // T must match the registered type (const and volatile are ignored in this comparison)
    // N must match the registered number of dimensions
    template <class T, int N , typename std::enable_if< std::is_const<T>::value , int >::type = 0 >
    Array<T,N,memSpace,styleC> get( std::string name ) const {
      // Make sure we have this name as an entry
      int id = find_entry_or_error( name );
      // Make sure it's the right type and dimensionality
      if (!validate_type<T>(id)) {
        std::cerr << "ERROR: Calling get() with name [" << name << "] with the wrong type"; endrun("");
      }
      if (!validate_dims<N>(id)) {
        std::cerr << "ERROR: Calling get() with name [" << name << "] with the wrong number of dimensions"; endrun("");
      }
      Array<T,N,memSpace,styleC> ret( name.c_str() , (T *) entries[id].ptr , entries[id].dims );
      return ret;
    }


    // Get a READ/WRITE YAKL array (styleC) for the entry of this name
    // If T is not const, then the dirty flag is set to true because it can be potentially written to
    // T must match the registered type (const and volatile are ignored in this comparison)
    // N must match the registered number of dimensions
    template <class T, int N , typename std::enable_if< ! std::is_const<T>::value , int >::type = 0 >
    Array<T,N,memSpace,styleC> get( std::string name ) {
      // Make sure we have this name as an entry
      int id = find_entry_or_error( name );
      entries[id].dirty = true;
      // Make sure it's the right type and dimensionality
      if (!validate_type<T>(id)) {
        std::cerr << "ERROR: Calling get() with name [" << name << "] with the wrong type"; endrun("");
      }
      if (!validate_dims<N>(id)) {
        std::cerr << "ERROR: Calling get() with name [" << name << "] with the wrong number of dimensions"; endrun("");
      }
      Array<T,N,memSpace,styleC> ret( name.c_str() , (T *) entries[id].ptr , entries[id].dims );
      return ret;
    }


    // Get a READ ONLY YAKL array (styleC) for the entry of this name
    // If T is not const, then the dirty flag is set to true because it can be potentially written to
    // T must match the registered type (const and volatile are ignored in this comparison)
    // First dimension is assumed to be the vertical index
    // All dimensions after first dimension are assumed to be horizontal indices that can be aggregated without
    //     regard to ordering. Fastest varying dimensions in the aggregated horizontal dimensions are maintained.
    template <class T, typename std::enable_if< std::is_const<T>::value , int>::type = 0 >
    Array<T,2,memSpace,styleC> get_lev_col( std::string name ) const {
      // Make sure we have this name as an entry
      int id = find_entry_or_error( name );
      // Make sure it's the right type
      if (!validate_type<T>(id)) {
        std::cerr << "ERROR: Calling get_lev_col() with name [" << name << "] with the wrong type"; endrun("");
      }
      if (!validate_dims_lev_col(id)) {
        std::cerr << "ERROR: Calling get_lev_col() with name [" << name << "], but the variable's number of " <<
                     "dimensions is not compatible. You need two or more dimensions in the variable to call this.";
        endrun("");
      }
      int nlev = entries[id].dims[0];
      int ncol = 1;
      for (int i=1; i < entries[id].dims.size(); i++) {
        ncol *= entries[id].dims[i];
      }
      Array<T,2,memSpace,styleC> ret( name.c_str() , (T *) entries[id].ptr , nlev , ncol );
      return ret;
    }


    // Get a READ/WRITE YAKL array (styleC) for the entry of this name
    // If T is not const, then the dirty flag is set to true because it can be potentially written to
    // T must match the registered type (const and volatile are ignored in this comparison)
    // First dimension is assumed to be the vertical index
    // All dimensions after first dimension are assumed to be horizontal indices that can be aggregated without
    //     regard to ordering. Fastest varying dimensions in the aggregated horizontal dimensions are maintained.
    template <class T, typename std::enable_if< ! std::is_const<T>::value , int>::type = 0 >
    Array<T,2,memSpace,styleC> get_lev_col( std::string name ) {
      // Make sure we have this name as an entry
      int id = find_entry_or_error( name );
      entries[id].dirty = true;
      // Make sure it's the right type
      validate_type<T>(id);
      validate_dims_lev_col(id);
      int nlev = entries[id].dims[0];
      int ncol = 1;
      for (int i=1; i < entries[id].dims.size(); i++) {
        ncol *= entries[id].dims[i];
      }
      Array<T,2,memSpace,styleC> ret( name.c_str() , (T *) entries[id].ptr , nlev , ncol );
      return ret;
    }


    // Get a READ ONLY YAKL array (styleC) for the entry of this name
    // If T is not const, then the dirty flag is set to true because it can be potentially written to
    // T must match the registered type (const and volatile are ignored in this comparison)
    // All dimensions are collapsed to a single dimension.
    // Fastest varying dimensions in the aggregated dimensions are maintained.
    template <class T, typename std::enable_if< std::is_const<T>::value , int>::type = 0 >
    Array<T,1,memSpace,styleC> get_collapsed( std::string name ) const {
      // Make sure we have this name as an entry
      int id = find_entry_or_error( name );
      // Make sure it's the right type
      validate_type<T>(id);
      int ncells = entries[id].dims[0];
      for (int i=1; i < entries[id].dims.size(); i++) {
        ncells *= entries[id].dims[i];
      }
      Array<T,1,memSpace,styleC> ret( name.c_str() , (T *) entries[id].ptr , ncells );
      return ret;
    }


    // Get a READ/WRITE YAKL array (styleC) for the entry of this name
    // If T is not const, then the dirty flag is set to true because it can be potentially written to
    // T must match the registered type (const and volatile are ignored in this comparison)
    // All dimensions are collapsed to a single dimension.
    // Fastest varying dimensions in the aggregated dimensions are maintained.
    template <class T, typename std::enable_if< ! std::is_const<T>::value , int>::type = 0 >
    Array<T,1,memSpace,styleC> get_collapsed( std::string name ) {
      // Make sure we have this name as an entry
      int id = find_entry_or_error( name );
      entries[id].dirty = true;
      // Make sure it's the right type
      validate_type<T>(id);
      int ncells = entries[id].dims[0];
      for (int i=1; i < entries[id].dims.size(); i++) {
        ncells *= entries[id].dims[i];
      }
      Array<T,1,memSpace,styleC> ret( name.c_str() , (T *) entries[id].ptr , ncells );
      return ret;
    }


    // Validate all numerical entries. positive-definite entries are validated to ensure no negative values
    // All floating point values are checked for infinities. All entries are checked for NaNs.
    // This is EXPENSIVE. All arrays are copied to the host, and the checks are performed on the host
    void validate_all( bool die_on_failed_check = false ) const {
      for (int id = 0; id < entries.size(); id++) { validate( entries[id].name , die_on_failed_check ); }
    }


    // Validate one entry. positive-definite entries are validated to ensure no negative values
    // All floating point values are checked for infinities. All entries are checked for NaNs.
    // This is EXPENSIVE. All arrays are copied to the host, and the checks are performed on the host
    void validate( std::string name , bool die_on_failed_check = false ) const {
      validate_nan(name,die_on_failed_check);
      validate_inf(name,die_on_failed_check);
      validate_pos(name,die_on_failed_check);
    }


    // Validate one entry for NaNs
    // This is EXPENSIVE. All arrays are copied to the host, and the checks are performed on the host
    void validate_nan( std::string name , bool die_on_failed_check = false ) const {
      bool die = die_on_failed_check;
      int id = find_entry_or_error(name);
      if      (entry_type_is_same<short int>             (id)) { validate_single_nan<short int const>             (name,die); }
      else if (entry_type_is_same<int>                   (id)) { validate_single_nan<int const>                   (name,die); }
      else if (entry_type_is_same<long int>              (id)) { validate_single_nan<long int const>              (name,die); }
      else if (entry_type_is_same<long long int>         (id)) { validate_single_nan<long long int const>         (name,die); }
      else if (entry_type_is_same<unsigned short int>    (id)) { validate_single_nan<unsigned short int const>    (name,die); }
      else if (entry_type_is_same<unsigned int>          (id)) { validate_single_nan<unsigned int const>          (name,die); }
      else if (entry_type_is_same<unsigned long int>     (id)) { validate_single_nan<unsigned long int const>     (name,die); }
      else if (entry_type_is_same<unsigned long long int>(id)) { validate_single_nan<unsigned long long int const>(name,die); }
      else if (entry_type_is_same<float>                 (id)) { validate_single_nan<float const>                 (name,die); }
      else if (entry_type_is_same<double>                (id)) { validate_single_nan<double const>                (name,die); }
      else if (entry_type_is_same<long double>           (id)) { validate_single_nan<long double const>           (name,die); }
    }


    // Validate one entry for infs
    // This is EXPENSIVE. All arrays are copied to the host, and the checks are performed on the host
    void validate_inf( std::string name , bool die_on_failed_check = false ) const {
      int id = find_entry_or_error(name);
      if      (entry_type_is_same<float>      (id)) { validate_single_inf<float const>      (name,die_on_failed_check); }
      else if (entry_type_is_same<double>     (id)) { validate_single_inf<double const>     (name,die_on_failed_check); }
      else if (entry_type_is_same<long double>(id)) { validate_single_inf<long double const>(name,die_on_failed_check); }
    }


    // Validate one entry for negative values
    // This is EXPENSIVE. All arrays are copied to the host, and the checks are performed on the host
    void validate_pos( std::string name , bool die_on_failed_check = false ) const {
      int id = find_entry_or_error(name);
      if      (entry_type_is_same<short int>    (id)) { validate_single_pos<short int const>    (name,die_on_failed_check); }
      else if (entry_type_is_same<int>          (id)) { validate_single_pos<int const>          (name,die_on_failed_check); }
      else if (entry_type_is_same<long int>     (id)) { validate_single_pos<long int const>     (name,die_on_failed_check); }
      else if (entry_type_is_same<long long int>(id)) { validate_single_pos<long long int const>(name,die_on_failed_check); }
      else if (entry_type_is_same<float>        (id)) { validate_single_pos<float const>        (name,die_on_failed_check); }
      else if (entry_type_is_same<double>       (id)) { validate_single_pos<double const>       (name,die_on_failed_check); }
      else if (entry_type_is_same<long double>  (id)) { validate_single_pos<long double const>  (name,die_on_failed_check); }
    }


    // INTERNAL USE: check one entry id for NaNs
    template <class T>
    void validate_single_nan(std::string name , bool die_on_failed_check = false) const {
      auto arr = get_collapsed<T>(name).createHostCopy();
      for (int i=0; i < arr.get_elem_count(); i++) {
        if ( std::isnan( arr(i) ) ) {
          std::cerr << "WARNING: NaN discovered in: " << name << " at global index: " << i << "\n";
          if (die_on_failed_check) endrun("");
        }
      }
    }


    // INTERNAL USE: check one entry id for infs
    template <class T>
    void validate_single_inf(std::string name , bool die_on_failed_check = false) const {
      auto arr = get_collapsed<T>(name).createHostCopy();
      for (int i=0; i < arr.get_elem_count(); i++) {
        if ( std::isinf( arr(i) ) ) {
          std::cerr << "WARNING: inf discovered in: " << name << " at global index: " << i << "\n";
          if (die_on_failed_check) endrun("");
        }
      }
    }


    // INTERNAL USE: check one entry id for negative values
    template <class T>
    void validate_single_pos(std::string name , bool die_on_failed_check = false) const {
      int id = find_entry_or_error( name );
      if (entries[id].positive) {
        auto arr = get_collapsed<T>(name).createHostCopy();
        for (int i=0; i < arr.get_elem_count(); i++) {
          if ( arr(i) < 0. ) {
            std::cerr << "WARNING: negative value discovered in positive-definite entry: " << name
                      << " at global index: " << i << "\n";
            if (die_on_failed_check) endrun("");
          }
        }
      }
    }


    // INTERNAL USE: Return the id of the named entry or -1 if it isn't found
    int find_entry( std::string name ) const {
      for (int i=0; i < entries.size(); i++) {
        if (entries[i].name == name) return i;
      }
      return -1;
    }


    // INTERNAL USE: Return the id of the named dimension or -1 if it isn't found
    int find_dimension( std::string name ) const {
      for (int i=0; i < dimensions.size(); i++) {
        if (dimensions[i].name == name) return i;
      }
      return -1;
    }


    // INTERNAL USE: Return the id of the named dimension or kill the run if it isn't found
    int find_entry_or_error( std::string name ) const {
      int id = find_entry( name );
      if (id >= 0) return id;
      std::cerr << "ERROR: Attempting to retrieve variable name [" << name << "], but it doesn't exist. ";
      endrun("");
      return -1;
    }


    // INTERNAL USE: Return the product of the vector of dimensions
    int get_data_size( std::vector<int> dims ) const {
      int size = 1;
      for (int i=0; i < dims.size(); i++) { size *= dims[i]; }
      return size;
    }


    // INTERNAL USE: Return the size of the named dimension or kill the run if it isn't found
    int get_dimension_size( std::string name ) const {
      int id = find_dimension( name );
      if (id == -1) {
        std::cerr << "ERROR: Attempting to get size of dimension name [" << name << "], but it doesn't exist. ";
        endrun("ERROR: Could not find dimension.");
      }
      return dimensions[id].len;
    }


    // INTERNAL USE: Return the C++ hash of this type. Ignore const and volatiles modifiers
    template <class T> size_t get_type_hash() const {
      return typeid(typename std::remove_cv<T>::type).hash_code();
    }


    // INTERNAL USE: Return whether the entry id's type is the same as the templated type
    template <class T> size_t entry_type_is_same(int id) const {
      return entries[id].type_hash == get_type_hash<T>();
    }


    // INTERNAL USE: End the run if the templated type is not the same as the entry id's type
    template <class T>
    bool validate_type(int id) const {
      if ( entries[id].type_hash != get_type_hash<T>() ) return false;
      return true;
    }


    // INTERNAL USE: End the run if the templated number of dimensions is not the same as the entry id's
    //     number of dimensions
    template <int N>
    bool validate_dims(int id) const {
      if ( N != entries[id].dims.size() ) return false;
      return true;
    }


    // INTERNAL USE: End the run if the entry id's of dimensions < 2
    bool validate_dims_lev_col(int id) const {
      if ( entries[id].dims.size() < 2 ) return false;
      return true;
    }


    // Deallocate all entries, and set the entries and dimensions to empty vectors. This is called by the destructor
    // Generally meat for internal use, but perhaps there are cases where the user might want to call this directly.
    void finalize() {
      yakl::fence();
      for (int i=0; i < entries.size(); i++) {
        deallocate( entries[i].ptr , entries[i].name.c_str() );
      }
      entries    = std::vector<Entry>();
      dimensions = std::vector<Dimension>();
    }


  };

  // Some useful typedefs for the user to use
  typedef DataManagerTemplate<yakl::memDevice> DataManager;
  typedef DataManagerTemplate<yakl::memHost> DataManagerHost;

}


